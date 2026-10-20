
"""
EDA Phase 2 - Task 02
Merge loan-level event tables and audit data quality.
"""

from pathlib import Path

import duckdb
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

YEARS = range(2006, 2023)

FILES = [
    DATA_DIR / f"02_loan_events_{year}.parquet"
    for year in YEARS
]

EXPECTED_COLUMNS = [
    "first_payment",
    "loan_id",
    "vintage",
    "monthly_records",
    "first_period",
    "last_period",
    "first_90_period",
    "zero_code",
    "zero_effective",
    "exit_reporting_period",
    "exit_group",
    "months_to_first_90",
    "ever_90",
]


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


# --------------------------------------------------
# 1. Verify input files and schemas
# --------------------------------------------------

missing = [str(p) for p in FILES if not p.exists()]

if missing:
    raise FileNotFoundError(
        f"Missing input files:\n" + "\n".join(missing)
    )

con = duckdb.connect()
con.execute("SET threads = 2")
con.execute("SET memory_limit = '1GB'")

for file in FILES:
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_path(file)})"
    ).fetchall()

    columns = [row[0] for row in schema]

    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema in {file.name}:\n{columns}"
        )

print("All 17 files and schemas verified.", flush=True)


# --------------------------------------------------
# 2. Create a unified DuckDB view
# --------------------------------------------------

paths = ", ".join(sql_path(p) for p in FILES)

con.execute(f"""
    CREATE VIEW events AS
    SELECT *
    FROM read_parquet([{paths}])
""")

check = con.execute("""
    SELECT
        COUNT(*) AS total_loans,
        COUNT(DISTINCT loan_id) AS unique_loans,
        COUNT(DISTINCT vintage) AS vintages
    FROM events
""").fetchone()

print("\nDataset validation:")
print(f"Total rows: {check[0]:,}")
print(f"Unique loans: {check[1]:,}")
print(f"Vintages: {check[2]}")

if check != (850000, 850000, 17):
    raise ValueError(
        "Unexpected row count, duplicate loan IDs, "
        "or missing vintages."
    )

vintage_counts = con.execute("""
    SELECT vintage, COUNT(*) AS loans
    FROM events
    GROUP BY vintage
    ORDER BY vintage
""").df()

if (
    vintage_counts["vintage"].tolist()
    != list(YEARS)
    or not vintage_counts["loans"].eq(50000).all()
):
    raise ValueError(
        "Expected exactly 50,000 loans per vintage."
    )


# --------------------------------------------------
# 3. Data quality audit
# --------------------------------------------------

quality = con.execute("""
    SELECT
        vintage,
        COUNT(*) AS loans,

        SUM(CASE
            WHEN monthly_records <= 0
                 OR first_period IS NULL
                 OR last_period IS NULL
                 OR first_period > last_period
            THEN 1 ELSE 0
        END) AS invalid_observation,

        SUM(CASE
            WHEN ever_90
                 AND (
                     first_90_period IS NULL
                     OR months_to_first_90 IS NULL
                 )
            THEN 1 ELSE 0
        END) AS missing_default_information,

        SUM(CASE
            WHEN first_90_period IS NOT NULL
                 AND (
                     first_90_period < first_period
                     OR first_90_period > last_period
                 )
            THEN 1 ELSE 0
        END) AS default_outside_observation,

        SUM(CASE
            WHEN months_to_first_90 <= 0
            THEN 1 ELSE 0
        END) AS invalid_default_timing,

        SUM(CASE
            WHEN zero_code <> ''
                 AND zero_effective = ''
            THEN 1 ELSE 0
        END) AS missing_exit_date

    FROM events
    GROUP BY vintage
    ORDER BY vintage
""").df()

quality.to_csv(
    OUT_DIR / "03_data_quality.csv",
    index=False
)

# Save records requiring closer inspection
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

    FROM events

    WHERE monthly_records <= 0
       OR first_period IS NULL
       OR last_period IS NULL
       OR first_period > last_period
       OR (
           ever_90 AND (
               first_90_period IS NULL
               OR months_to_first_90 IS NULL
           )
       )
       OR (
           first_90_period IS NOT NULL
           AND (
               first_90_period < first_period
               OR first_90_period > last_period
           )
       )
       OR months_to_first_90 <= 0
       OR (
           zero_code <> ''
           AND zero_effective = ''
       )

    ORDER BY vintage, loan_id
""").df()

anomalies.to_csv(
    OUT_DIR / "03_anomalous_events.csv",
    index=False
)


# --------------------------------------------------
# 4. Cross-vintage statistics
# --------------------------------------------------

summary = con.execute("""
    SELECT
        vintage,

        COUNT(*) AS loans,

        SUM(monthly_records) AS performance_rows,

        SUM(CASE WHEN ever_90
            THEN 1 ELSE 0 END) AS ever_90_loans,

        ROUND(
            100.0 * AVG(
                CASE WHEN ever_90 THEN 1.0 ELSE 0.0 END
            ), 2
        ) AS observed_ever90_pct,

        SUM(CASE
            WHEN zero_code <> ''
            THEN 1 ELSE 0
        END) AS exited_loans,

        SUM(CASE
            WHEN zero_code = ''
            THEN 1 ELSE 0
        END) AS no_reported_exit,

        MEDIAN(
            CASE
                WHEN months_to_first_90 >= 1
                THEN months_to_first_90
                ELSE NULL
            END
        ) AS median_month_to_first90,

        MAX(last_period) AS latest_period

    FROM events
    GROUP BY vintage
    ORDER BY vintage
""").df()

summary.to_csv(
    OUT_DIR / "03_vintage_summary.csv",
    index=False
)

exit_summary = con.execute("""
    SELECT
        vintage,
        exit_group,
        COUNT(*) AS loans

    FROM events
    GROUP BY vintage, exit_group
    ORDER BY vintage, exit_group
""").df()

exit_summary.to_csv(
    OUT_DIR / "03_exit_summary.csv",
    index=False
)


# --------------------------------------------------
# 5. Visualizations
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.bar(
    summary["vintage"],
    summary["observed_ever90_pct"]
)

ax.set(
    xlabel="Origination Vintage",
    ylabel="Observed Ever 90+ DPD (%)",
    title="Observed Severe Delinquency by Vintage"
)

ax.set_xticks(summary["vintage"])
ax.tick_params(axis="x", rotation=45)

ax.text(
    0.01, -0.28,
    "Descriptive only: vintages have different "
    "observation windows. Not 12-month PD.",
    transform=ax.transAxes,
    fontsize=9
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "03_ever90_by_vintage.png",
    dpi=160,
    bbox_inches="tight"
)
plt.close(fig)


exit_matrix = exit_summary.pivot(
    index="vintage",
    columns="exit_group",
    values="loans"
).fillna(0)

exit_shares = exit_matrix.div(
    exit_matrix.sum(axis=1),
    axis=0
) * 100

ax = exit_shares.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 5),
    width=0.85
)

ax.set(
    xlabel="Origination Vintage",
    ylabel="Share of Loans (%)",
    title="Loan Exit Status by Vintage"
)

ax.legend(
    title="Exit Group",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(
    FIG_DIR / "03_exit_mix_by_vintage.png",
    dpi=160,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 6. Save consolidated Parquet
# --------------------------------------------------

merged_file = (
    DATA_DIR / "03_loan_events_2006_2022.parquet"
)

if merged_file.exists():
    raise FileExistsError(
        f"{merged_file} already exists. "
        "Review it before overwriting."
    )

print("\nWriting consolidated Parquet...", flush=True)

con.execute(f"""
    COPY (
        SELECT * FROM events
    )
    TO {sql_path(merged_file)}
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== Cross-Vintage Summary ===")
print(summary.to_string(index=False))

print("\n=== Data Quality ===")
print(quality.to_string(index=False))

print(f"\nAnomalous loans: {len(anomalies):,}")
print(f"Merged file: {merged_file}")
print("\nEDA Task 02 completed.")

con.close()
