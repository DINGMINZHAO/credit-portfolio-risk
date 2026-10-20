
"""
EDA Phase 2 - Task 06
Fixed-Horizon Default Analysis

Objective:
Classify first observed 90+ DPD events within
12, 24 and 36 months of first scheduled payment.

Results are descriptive, NOT adjusted PD estimates.
"""

from pathlib import Path

import duckdb
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. Paths and environment
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data/processed"
    / "03_loan_events_2006_2022.parquet"
)

GAP_FILE = ROOT / "outputs/05_missing_months.csv"

OUT = ROOT / "outputs"
FIG = OUT / "figures"
PROCESSED = ROOT / "data/processed"

FIG.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if not GAP_FILE.exists():
    raise FileNotFoundError(GAP_FILE)

con = duckdb.connect()

con.execute("SET memory_limit = '1GB'")
con.execute("SET threads = 2")

source_sql = str(SOURCE).replace("'", "''")


# ==========================================
# 2. Identify loans with missing months
# ==========================================

gaps = pd.read_csv(
    GAP_FILE,
    dtype={"loan_id": str}
)

gap_ids = gaps[["loan_id"]].drop_duplicates()

con.register("gap_ids", gap_ids)


# ==========================================
# 3. Convert dates to month indices
# ==========================================

# Example:
# 202001 -> 2020 * 12 + 1
# 202002 -> 2020 * 12 + 2
#
# Difference = 1 month

con.execute(f"""
CREATE VIEW timeline AS

WITH raw AS (

    SELECT
        *,

        TRY_CAST(
            SUBSTR(first_payment, 1, 4)
            AS INTEGER
        ) AS fp_year,

        TRY_CAST(
            SUBSTR(first_payment, 5, 2)
            AS INTEGER
        ) AS fp_month,

        TRY_CAST(
            NULLIF(zero_effective, '')
            AS INTEGER
        ) AS exit_ym

    FROM read_parquet('{source_sql}')

)

SELECT
    *,

    fp_year * 12 + fp_month AS fp_m,

    CAST(FLOOR(first_period / 100)
        AS INTEGER) * 12
        + first_period % 100 AS first_m,

    CAST(FLOOR(last_period / 100)
        AS INTEGER) * 12
        + last_period % 100 AS last_m,

    CAST(FLOOR(first_90_period / 100)
        AS INTEGER) * 12
        + first_90_period % 100 AS default_m,

    CAST(FLOOR(exit_ym / 100)
        AS INTEGER) * 12
        + exit_ym % 100 AS exit_m

FROM raw
""")


# ==========================================
# 4. Select baseline cohort
# ==========================================

# We exclude loans without sufficiently
# reliable observation from Month 1.
#
# Exclusion does not mean the loan is bad data.

con.execute("""
CREATE VIEW cohort AS

SELECT
    t.*,

    CASE
        WHEN fp_year IS NULL
          OR fp_month NOT BETWEEN 1 AND 12
          OR first_m IS NULL
          OR last_m IS NULL
            THEN 'Invalid date'

        WHEN fp_year NOT BETWEEN vintage
                             AND vintage + 1
            THEN 'Special first payment date'

        WHEN first_m > fp_m
            THEN 'Delayed observation'

        WHEN first_m < fp_m - 1
            THEN 'Unusual early observation'

        WHEN default_m < fp_m
            THEN 'Default before time origin'

        WHEN exit_m < fp_m
            THEN 'Exit before time origin'

        WHEN default_m > exit_m
            THEN 'Invalid event order'

        WHEN zero_code <> ''
             AND exit_m IS NULL
            THEN 'Missing exit date'

        WHEN g.loan_id IS NOT NULL
            THEN 'Monthly observation gap'

        ELSE 'Eligible'

    END AS eligibility

FROM timeline t

LEFT JOIN gap_ids g
ON t.loan_id = g.loan_id
""")


# ==========================================
# 5. Classify outcomes at each horizon
# ==========================================

# Month 1 = first scheduled payment month.
#
# 12M: fp_m ... fp_m + 11
# 24M: fp_m ... fp_m + 23
# 36M: fp_m ... fp_m + 35

con.execute("""
CREATE VIEW horizon_events AS

WITH expanded AS (

    SELECT
        c.*,
        h.horizon,
        fp_m + h.horizon - 1 AS horizon_end

    FROM cohort c

    CROSS JOIN (
        VALUES (12), (24), (36)
    ) AS h(horizon)

)

SELECT
    loan_id,
    vintage,
    horizon,
    eligibility,
    exit_group,

    CASE
        WHEN eligibility <> 'Eligible'
            THEN 'Excluded'

        WHEN default_m BETWEEN fp_m AND horizon_end
            THEN 'First 90+ DPD'

        WHEN exit_m BETWEEN fp_m AND horizon_end
            THEN 'Exit without prior 90+'

        WHEN last_m >= horizon_end
            THEN 'No 90+ through H'

        ELSE 'Incomplete follow-up'

    END AS outcome

FROM expanded
""")


# ==========================================
# 6. Count loans by vintage and horizon
# ==========================================

summary = con.execute("""
SELECT
    vintage,
    horizon,

    COUNT(*) AS total_loans,

    COUNT(*) FILTER (
        WHERE eligibility = 'Eligible'
    ) AS eligible,

    COUNT(*) FILTER (
        WHERE outcome = 'First 90+ DPD'
    ) AS first90,

    COUNT(*) FILTER (
        WHERE outcome = 'Exit without prior 90+'
    ) AS exit_before90,

    COUNT(*) FILTER (
        WHERE outcome = 'No 90+ through H'
    ) AS no90,

    COUNT(*) FILTER (
        WHERE outcome = 'Incomplete follow-up'
    ) AS incomplete,

    COUNT(*) FILTER (
        WHERE outcome = 'Excluded'
    ) AS excluded

FROM horizon_events

GROUP BY vintage, horizon

ORDER BY vintage, horizon
""").df()


# ==========================================
# 7. Validate accounting
# ==========================================

classified = (
    summary["first90"]
    + summary["exit_before90"]
    + summary["no90"]
    + summary["incomplete"]
)

if not (classified == summary["eligible"]).all():
    raise ValueError("Eligible loan counts do not reconcile.")

if not (
    summary["eligible"] + summary["excluded"]
    == summary["total_loans"]
).all():
    raise ValueError("Total loan counts do not reconcile.")


# ==========================================
# 8. Calculate descriptive event shares
# ==========================================

summary["observed_first90_pct"] = (
    100 * summary["first90"] / summary["eligible"]
).round(3)

summary["incomplete_pct"] = (
    100 * summary["incomplete"] / summary["eligible"]
).round(3)

summary.to_csv(
    OUT / "07_horizon_summary.csv",
    index=False
)


# ==========================================
# 9. Record exclusions and exit types
# ==========================================

eligibility = con.execute("""
SELECT
    vintage,
    eligibility,
    COUNT(*) AS loans

FROM cohort

GROUP BY vintage, eligibility
ORDER BY vintage, eligibility
""").df()

eligibility.to_csv(
    OUT / "07_cohort_eligibility.csv",
    index=False
)

exit_summary = con.execute("""
SELECT
    vintage,
    horizon,
    exit_group,
    COUNT(*) AS loans

FROM horizon_events

WHERE outcome = 'Exit without prior 90+'

GROUP BY vintage, horizon, exit_group
ORDER BY vintage, horizon, exit_group
""").df()

exit_summary.to_csv(
    OUT / "07_exit_before90.csv",
    index=False
)


# Save loan-level outcomes for later analysis
output_parquet = (
    PROCESSED / "07_horizon_events.parquet"
)

output_sql = str(output_parquet).replace("'", "''")

con.execute(f"""
COPY (
    SELECT *
    FROM horizon_events
)
TO '{output_sql}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")


# ==========================================
# 10. Figure 1: Default events by vintage
# ==========================================

fig, ax = plt.subplots(figsize=(10, 5))

for horizon in [12, 24, 36]:

    subset = summary[
        summary["horizon"] == horizon
    ]

    ax.plot(
        subset["vintage"],
        subset["observed_first90_pct"],
        marker="o",
        label=f"{horizon}M"
    )

ax.set(
    title="First 90+ DPD by Origination Vintage",
    xlabel="Origination Vintage",
    ylabel="Observed First 90+ Share (%)"
)

ax.set_xticks(
    sorted(summary["vintage"].unique())
)

ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.2)
ax.legend(title="Horizon")

fig.tight_layout()

fig.savefig(
    FIG / "07_first90_by_vintage.png",
    dpi=160
)

plt.close(fig)


# ==========================================
# 11. Figure 2: 36M outcome composition
# ==========================================

d36 = summary[
    summary["horizon"] == 36
].copy()

columns = [
    "first90",
    "exit_before90",
    "no90",
    "incomplete"
]

labels = [
    "First 90+ DPD",
    "Exit before 90+",
    "No 90+ through 36M",
    "Incomplete follow-up"
]

plot_data = d36.set_index("vintage")[columns]

plot_shares = (
    plot_data.div(
        d36.set_index("vintage")["eligible"],
        axis=0
    ) * 100
)

plot_shares.columns = labels

ax = plot_shares.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 5),
    width=0.85
)

ax.set(
    title="36M Loan Outcomes by Vintage",
    xlabel="Origination Vintage",
    ylabel="Share of Eligible Loans (%)"
)

ax.set_ylim(0, 100)

ax.legend(
    title="Outcome",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIG / "07_36m_outcome_mix.png",
    dpi=160,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. Terminal output
# ==========================================

print("\n=== Cohort Eligibility ===")

print(
    eligibility.groupby("eligibility")["loans"]
    .sum()
    .sort_values(ascending=False)
    .to_string()
)

print("\n=== Observed First 90+ Share (%) ===")

wide = summary.pivot(
    index="vintage",
    columns="horizon",
    values="observed_first90_pct"
)

wide.columns = [
    f"{int(c)}M" for c in wide.columns
]

print(wide.round(2).to_string())

print("\n=== Incomplete Follow-up ===")

incomplete = summary.pivot(
    index="vintage",
    columns="horizon",
    values="incomplete"
)

print(incomplete.to_string())

print("\n=== Portfolio Event Counts ===")

print(
    summary.groupby("horizon")[
        [
            "eligible",
            "first90",
            "exit_before90",
            "no90",
            "incomplete"
        ]
    ].sum().to_string()
)

print("\nTask 06 completed.")

con.close()
