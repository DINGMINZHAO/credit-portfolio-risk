
"""
EDA Phase 2 - Task 03
Observation Window & Censoring Audit

Input:
    03_loan_events_2006_2022.parquet

Outputs:
    04_observation_audit.csv
    04_timing_anomalies.csv
    04_early_stop_sample.csv
    04_observation_status.png
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------
# Configuration
# ----------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "processed"
    / "03_loan_events_2006_2022.parquet"
)

OUT = ROOT / "outputs"
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

con = duckdb.connect()

con.execute("SET memory_limit = '1GB'")
con.execute("SET threads = 2")

source_sql = str(SOURCE).replace("'", "''")

con.execute(f"""
    CREATE VIEW events AS
    SELECT *
    FROM read_parquet('{source_sql}')
""")


# ----------------------------------------------
# 1. Dataset reporting cutoff
# ----------------------------------------------

cutoff = con.execute("""
    SELECT MAX(last_period)
    FROM events
""").fetchone()[0]

print(f"\nDataset cutoff: {cutoff}")


# ----------------------------------------------
# 2. Observation status and calendar span
# ----------------------------------------------

con.execute(f"""
    CREATE VIEW audit AS

    SELECT
        *,

        (
            (
                CAST(FLOOR(last_period / 100) AS INTEGER)
                -
                CAST(FLOOR(first_period / 100) AS INTEGER)
            ) * 12
            +
            (last_period % 100)
            -
            (first_period % 100)
            + 1
        ) AS calendar_span,

        CASE
            WHEN zero_code <> ''
                THEN 'Recorded exit'

            WHEN last_period = {int(cutoff)}
                THEN 'Observed at cutoff'

            ELSE 'Early stop (no exit)'
        END AS observation_status

    FROM events
""")


# ----------------------------------------------
# 3. Vintage-level audit
# ----------------------------------------------

summary = con.execute("""
    SELECT
        vintage,

        COUNT(*) AS loans,

        SUM(CASE
            WHEN observation_status = 'Recorded exit'
            THEN 1 ELSE 0
        END) AS recorded_exit,

        SUM(CASE
            WHEN observation_status = 'Observed at cutoff'
            THEN 1 ELSE 0
        END) AS at_cutoff,

        SUM(CASE
            WHEN observation_status = 'Early stop (no exit)'
            THEN 1 ELSE 0
        END) AS early_stop,

        SUM(CASE
            WHEN monthly_records < calendar_span
            THEN 1 ELSE 0
        END) AS fewer_records_than_span,

        SUM(CASE
            WHEN monthly_records > calendar_span
            THEN 1 ELSE 0
        END) AS more_records_than_span,

        SUM(CASE
            WHEN months_to_first_90 <= 0
            THEN 1 ELSE 0
        END) AS invalid_default_timing,

        ROUND(AVG(calendar_span), 1)
            AS avg_observed_months

    FROM audit

    GROUP BY vintage
    ORDER BY vintage
""").df()

summary.to_csv(
    OUT / "04_observation_audit.csv",
    index=False
)


# ----------------------------------------------
# 4. Inspect timing anomalies
# ----------------------------------------------

anomalies = con.execute("""
    SELECT
        loan_id,
        vintage,
        first_payment,
        first_period,
        last_period,
        first_90_period,
        months_to_first_90,
        zero_code,
        zero_effective

    FROM audit

    WHERE months_to_first_90 <= 0

    ORDER BY vintage, loan_id
""").df()

anomalies.to_csv(
    OUT / "04_timing_anomalies.csv",
    index=False
)


# ----------------------------------------------
# 5. Sample unexplained early stops
# ----------------------------------------------

early_stops = con.execute("""
    SELECT
        loan_id,
        vintage,
        first_period,
        last_period,
        monthly_records,
        calendar_span,
        ever_90,
        first_90_period

    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY vintage
                ORDER BY last_period, loan_id
            ) AS rn

        FROM audit

        WHERE observation_status = 'Early stop (no exit)'
    )

    WHERE rn <= 5

    ORDER BY vintage, last_period
""").df()

early_stops.to_csv(
    OUT / "04_early_stop_sample.csv",
    index=False
)


# ----------------------------------------------
# 6. Visualization
# ----------------------------------------------

plot_columns = [
    "recorded_exit",
    "at_cutoff",
    "early_stop",
]

plot_labels = [
    "Recorded exit",
    "Observed at cutoff",
    "Early stop (no exit)",
]

plot_data = summary.set_index("vintage")[plot_columns]

plot_shares = (
    plot_data.div(
        summary.set_index("vintage")["loans"],
        axis=0
    ) * 100
)

ax = plot_shares.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 5),
    width=0.85
)

ax.set(
    xlabel="Origination Vintage",
    ylabel="Share of Loans (%)",
    title="Observation Status by Vintage"
)

ax.set_ylim(0, 100)

ax.legend(
    plot_labels,
    title="Observation Status",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIG / "04_observation_status.png",
    dpi=160,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------
# 7. Terminal summary
# ----------------------------------------------

print("\n=== Observation Status ===")

print(
    summary[
        [
            "vintage",
            "recorded_exit",
            "at_cutoff",
            "early_stop",
        ]
    ].to_string(index=False)
)

print("\n=== Monthly Coverage ===")

print(
    summary[
        [
            "vintage",
            "fewer_records_than_span",
            "more_records_than_span",
            "invalid_default_timing",
            "avg_observed_months",
        ]
    ].to_string(index=False)
)

print("\n=== Portfolio Totals ===")

for col in [
    "recorded_exit",
    "at_cutoff",
    "early_stop",
    "fewer_records_than_span",
    "more_records_than_span",
]:
    print(f"{col}: {int(summary[col].sum()):,}")

print("\n=== Timing Anomalies ===")
print(anomalies.to_string(index=False))

print(
    f"\nEarly-stop sample saved: "
    f"{len(early_stops)} loans"
)

print("\nTask 03 completed.")

con.close()
