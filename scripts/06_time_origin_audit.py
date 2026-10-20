
"""
EDA Phase 2 - Task 05
Time Origin & Event Order Audit

Inputs:
  data/processed/03_loan_events_2006_2022.parquet
  outputs/05_missing_months.csv

No changes to existing datasets.
"""

from pathlib import Path

import duckdb
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/03_loan_events_2006_2022.parquet"
GAPS = ROOT / "outputs/05_missing_months.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

FIG.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists() or not GAPS.exists():
    raise FileNotFoundError("Required input files missing.")

con = duckdb.connect()
con.execute("SET memory_limit = '1GB'")
con.execute("SET threads = 2")

path_sql = str(SOURCE).replace("'", "''")


# --------------------------------------------------
# 1. Convert YYYYMM dates into month indices
# --------------------------------------------------

con.execute(f"""
CREATE VIEW timeline AS

WITH parsed AS (
    SELECT
        loan_id,
        vintage,
        first_payment,
        first_period,
        last_period,
        first_90_period,
        zero_code,
        zero_effective,

        (
            TRY_CAST(SUBSTR(first_payment, 1, 4) AS INTEGER) * 12
            + TRY_CAST(SUBSTR(first_payment, 5, 2) AS INTEGER)
        ) AS fp_m,

        (
            CAST(FLOOR(first_period / 100) AS INTEGER) * 12
            + first_period % 100
        ) AS first_m,

        (
            CAST(FLOOR(last_period / 100) AS INTEGER) * 12
            + last_period % 100
        ) AS last_m,

        (
            CAST(FLOOR(first_90_period / 100) AS INTEGER) * 12
            + first_90_period % 100
        ) AS first90_m,

        TRY_CAST(
            NULLIF(zero_effective, '') AS INTEGER
        ) AS exit_ym

    FROM read_parquet('{path_sql}')
)

SELECT
    *,
    first_m - fp_m AS report_lag,

    (
        CAST(FLOOR(exit_ym / 100) AS INTEGER) * 12
        + exit_ym % 100
    ) AS exit_m

FROM parsed
""")


# --------------------------------------------------
# 2. First reporting month vs first payment month
# --------------------------------------------------

summary = con.execute("""
SELECT
    vintage,
    COUNT(*) AS loans,

    SUM(CASE WHEN report_lag <= -2
        THEN 1 ELSE 0 END) AS earlier_2plus,

    SUM(CASE WHEN report_lag = -1
        THEN 1 ELSE 0 END) AS earlier_1,

    SUM(CASE WHEN report_lag = 0
        THEN 1 ELSE 0 END) AS same_month,

    SUM(CASE WHEN report_lag = 1
        THEN 1 ELSE 0 END) AS later_1,

    SUM(CASE WHEN report_lag >= 2
        THEN 1 ELSE 0 END) AS later_2plus,

    SUM(CASE WHEN report_lag IS NULL
        THEN 1 ELSE 0 END) AS invalid_dates,

    SUM(CASE WHEN first90_m < fp_m
        THEN 1 ELSE 0 END) AS default_before_first_payment,

    SUM(CASE WHEN
        TRY_CAST(SUBSTR(first_payment, 1, 4) AS INTEGER)
            > vintage + 1
        THEN 1 ELSE 0
    END) AS first_payment_late_vintage

FROM timeline
GROUP BY vintage
ORDER BY vintage
""").df()

summary.to_csv(
    OUT / "06_time_origin_summary.csv",
    index=False
)


# --------------------------------------------------
# 3. Check ordering of credit and exit events
# --------------------------------------------------

event_order = con.execute("""
SELECT
    COUNT(*) AS total_loans,

    SUM(CASE WHEN report_lag > 0
        THEN 1 ELSE 0 END) AS reporting_after_first_payment,

    SUM(CASE WHEN report_lag < -1
        THEN 1 ELSE 0 END) AS reporting_early_2plus,

    SUM(CASE WHEN first90_m < fp_m
        THEN 1 ELSE 0 END) AS default_before_first_payment,

    SUM(CASE WHEN exit_m < first_m
        THEN 1 ELSE 0 END) AS exit_before_first_report,

    SUM(CASE WHEN exit_m > last_m
        THEN 1 ELSE 0 END) AS exit_after_last_report,

    SUM(CASE WHEN first90_m > exit_m
        THEN 1 ELSE 0 END) AS default_after_exit,

    SUM(CASE WHEN first90_m = exit_m
        THEN 1 ELSE 0 END) AS default_same_month_as_exit

FROM timeline
""").df()

event_order.to_csv(
    OUT / "06_event_order_summary.csv",
    index=False
)


# --------------------------------------------------
# 4. Export unusual timelines for review
# --------------------------------------------------

unusual = con.execute("""
SELECT
    loan_id,
    vintage,
    first_payment,
    first_period,
    last_period,
    first_90_period,
    zero_code,
    zero_effective,
    report_lag

FROM timeline

WHERE ABS(report_lag) > 12
   OR first90_m < fp_m
   OR exit_m < first_m
   OR exit_m > last_m
   OR first90_m > exit_m

ORDER BY vintage, loan_id
""").df()

unusual.to_csv(
    OUT / "06_unusual_timelines.csv",
    index=False
)


# --------------------------------------------------
# 5. Examine missing months by loan age
# --------------------------------------------------

gaps = pd.read_csv(
    GAPS,
    dtype={
        "loan_id": str,
        "missing_period": str
    }
)

gaps["missing_ym"] = pd.to_numeric(
    gaps["missing_period"],
    errors="raise"
).astype("int64")

gaps["gap_m"] = (
    gaps["missing_ym"] // 100 * 12
    + gaps["missing_ym"] % 100
)

con.register(
    "gap_list",
    gaps[["loan_id", "gap_m"]]
)

gap_context = con.execute("""
SELECT
    g.loan_id,
    g.gap_m,
    t.vintage,
    t.fp_m,
    t.first90_m,
    t.exit_m,

    g.gap_m - t.fp_m + 1 AS gap_age

FROM gap_list g
JOIN timeline t
ON g.loan_id = t.loan_id
""").df()

gap_results = []

for horizon in [12, 24, 36]:

    within = gap_context["gap_age"].between(
        1, horizon
    )

    before_default = (
        gap_context["first90_m"].isna()
        | (
            gap_context["gap_m"]
            < gap_context["first90_m"]
        )
    )

    before_exit = (
        gap_context["exit_m"].isna()
        | (
            gap_context["gap_m"]
            < gap_context["exit_m"]
        )
    )

    relevant = within & before_default & before_exit

    gap_results.append({
        "horizon_months": horizon,
        "missing_loan_months": int(within.sum()),
        "affected_loans": int(
            gap_context.loc[within, "loan_id"].nunique()
        ),
        "missing_before_events": int(relevant.sum()),
    })

gap_summary = pd.DataFrame(gap_results)

gap_summary.to_csv(
    OUT / "06_missing_months_by_horizon.csv",
    index=False
)


# --------------------------------------------------
# 6. Visualization
# --------------------------------------------------

columns = [
    "earlier_2plus",
    "earlier_1",
    "same_month",
    "later_1",
    "later_2plus",
    "invalid_dates"
]

labels = [
    "At least 2 months earlier",
    "1 month earlier",
    "Same month",
    "1 month later",
    "At least 2 months later",
    "Invalid date"
]

shares = summary.set_index("vintage")[columns]

shares = shares.div(
    summary.set_index("vintage")["loans"],
    axis=0
) * 100

ax = shares.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 5),
    width=0.85
)

ax.set(
    xlabel="Origination Vintage",
    ylabel="Share of Loans (%)",
    title="First Reporting Month Relative to First Payment"
)

ax.set_ylim(0, 100)

ax.legend(
    labels,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIG / "06_first_report_timing.png",
    dpi=160,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 7. Print concise report
# --------------------------------------------------

print("\n=== First Reporting Timing ===")

print(
    summary[
        [
            "vintage",
            "earlier_2plus",
            "earlier_1",
            "same_month",
            "later_1",
            "later_2plus",
        ]
    ].to_string(index=False)
)

print("\n=== Event Order ===")

print(
    event_order.T.to_string(header=False)
)

print("\n=== Missing Months by Horizon ===")

print(gap_summary.to_string(index=False))

print(
    f"\nUnusual timelines for review: "
    f"{len(unusual):,}"
)

print("\nTask 05 completed.")

con.close()
