
"""
EDA Phase 2 - Task 04
Event Time Validation

Audit:
1. Monthly record gaps
2. Duplicate reporting months
3. Invalid default timing
4. First Payment Date consistency

Reads only the relevant annual ZIP files.
Does not modify existing datasets.
"""

from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict

import duckdb
import pandas as pd


# ==============================================
# Configuration
# ==============================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

SOURCE = (
    DATA / "processed"
    / "03_loan_events_2006_2022.parquet"
)

CHUNK_SIZE = 150_000

OUTPUTS.mkdir(parents=True, exist_ok=True)


# ==============================================
# Helper functions
# ==============================================

def month_serial(value):
    """Convert YYYYMM into a continuous month index."""

    s = str(value).strip()

    if len(s) != 6 or not s.isdecimal():
        return None

    year = int(s[:4])
    month = int(s[4:])

    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        return None

    return year * 12 + month - 1


def to_yyyymm(serial):
    year, month = divmod(serial, 12)
    return f"{year:04d}{month + 1:02d}"


def is_severe_delinquency(status):
    """Match the original extraction rule: numeric 03-99."""

    try:
        value = float(status)
        return 3 <= value <= 99
    except (ValueError, TypeError):
        return False


# ==============================================
# 1. Identify suspicious loans
# ==============================================

con = duckdb.connect()

con.execute("SET memory_limit = '1GB'")
con.execute("SET threads = 2")

flagged = con.execute("""
    WITH base AS (
        SELECT
            *,

            (
                (
                    CAST(FLOOR(last_period / 100) AS INTEGER)
                    -
                    CAST(FLOOR(first_period / 100) AS INTEGER)
                ) * 12
                +
                last_period % 100
                -
                first_period % 100
                + 1
            ) AS calendar_span

        FROM read_parquet(?)
    )

    SELECT
        loan_id,
        vintage,
        first_payment,
        first_period,
        last_period,
        monthly_records,
        first_90_period,
        months_to_first_90,
        calendar_span,

        monthly_records <> calendar_span
            AS coverage_flag,

        COALESCE(
            months_to_first_90 <= 0,
            FALSE
        ) AS timing_flag

    FROM base

    WHERE monthly_records <> calendar_span
       OR months_to_first_90 <= 0

    ORDER BY vintage, loan_id
""", [str(SOURCE)]).df()


# Check First Payment Date against vintage
population = con.execute("""
    SELECT
        vintage,
        COUNT(*) AS loans,

        SUM(
            CASE
                WHEN TRY_CAST(
                    SUBSTR(first_payment, 1, 4)
                    AS INTEGER
                ) > vintage + 1
                THEN 1 ELSE 0
            END
        ) AS first_payment_more_than_one_year_later,

        SUM(
            CASE
                WHEN TRY_CAST(
                    SUBSTR(first_payment, 1, 4)
                    AS INTEGER
                ) < vintage
                THEN 1 ELSE 0
            END
        ) AS first_payment_before_vintage

    FROM read_parquet(?)

    GROUP BY vintage
    ORDER BY vintage
""", [str(SOURCE)]).df()

population.to_csv(
    OUTPUTS / "05_first_payment_vintage_check.csv",
    index=False
)

con.close()


print("\n=== Flagged Loans ===")

print(f"Unique loans: {len(flagged):,}")

print(
    f"Coverage mismatches: "
    f"{int(flagged['coverage_flag'].sum()):,}"
)

print(
    f"Timing anomalies: "
    f"{int(flagged['timing_flag'].sum()):,}"
)


# ==============================================
# 2. Retrieve original monthly records
# ==============================================

raw_records = defaultdict(list)
raw_first_payment = {}

years = sorted(flagged["vintage"].unique())

for year in years:

    year = int(year)

    target_ids = set(
        flagged.loc[
            flagged["vintage"] == year,
            "loan_id"
        ]
    )

    archive = (
        DATA / "raw" / f"sample_{year}.zip"
    )

    print(
        f"\nChecking {year}: "
        f"{len(target_ids)} loans...",
        flush=True
    )

    with ZipFile(archive) as zf:

        # Read origination dates
        with zf.open(
            f"sample_orig_{year}.txt"
        ) as f:

            orig = pd.read_csv(
                f,
                sep="|",
                header=None,
                usecols=[1, 19],
                dtype=str,
                keep_default_na=False
            )

            orig.columns = [
                "first_payment",
                "loan_id"
            ]

            orig = orig.apply(
                lambda col: col.str.strip()
            )

            selected = orig[
                orig["loan_id"].isin(target_ids)
            ]

            raw_first_payment.update(
                zip(
                    selected["loan_id"],
                    selected["first_payment"]
                )
            )

        # Stream performance records
        with zf.open(
            f"sample_perf_{year}.txt"
        ) as f:

            reader = pd.read_csv(
                f,
                sep="|",
                header=None,
                usecols=[0, 1, 3, 8, 9],
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE
            )

            for chunk in reader:

                selected = chunk[
                    chunk[0].isin(target_ids)
                ]

                for row in selected.itertuples(
                    index=False, name=None
                ):

                    loan_id, period, dpd, code, exit_date = (
                        str(x).strip() for x in row
                    )

                    raw_records[loan_id].append(
                        (period, dpd, code, exit_date)
                    )


# ==============================================
# 3. Reconstruct event timelines
# ==============================================

audit_rows = []
missing_rows = []
duplicate_rows = []
timing_history = []

timing_ids = set(
    flagged.loc[
        flagged["timing_flag"], "loan_id"
    ]
)

for loan in flagged.itertuples(index=False):

    records = raw_records[loan.loan_id]

    month_ids = [
        month_serial(period)
        for period, _, _, _ in records
    ]

    valid_months = [
        m for m in month_ids if m is not None
    ]

    counts = Counter(valid_months)

    if valid_months:
        full_range = set(
            range(
                min(valid_months),
                max(valid_months) + 1
            )
        )

        missing = sorted(
            full_range - set(valid_months)
        )
    else:
        missing = []

    duplicates = [
        (m, count)
        for m, count in sorted(counts.items())
        if count > 1
    ]

    # Save each missing month
    for m in missing:
        missing_rows.append({
            "loan_id": loan.loan_id,
            "vintage": loan.vintage,
            "missing_period": to_yyyymm(m)
        })

    # Save duplicate months
    for m, count in duplicates:
        duplicate_rows.append({
            "loan_id": loan.loan_id,
            "vintage": loan.vintage,
            "duplicate_period": to_yyyymm(m),
            "count": count
        })

    # Recalculate first observed 90+ DPD
    default_months = [
        m
        for (_, dpd, _, _), m
        in zip(records, month_ids)
        if m is not None
        and is_severe_delinquency(dpd)
    ]

    first_90_raw = (
        to_yyyymm(min(default_months))
        if default_months else ""
    )

    first_90_saved = (
        str(int(loan.first_90_period))
        if pd.notna(loan.first_90_period)
        else ""
    )

    first_period_raw = (
        to_yyyymm(min(valid_months))
        if valid_months else ""
    )

    last_period_raw = (
        to_yyyymm(max(valid_months))
        if valid_months else ""
    )

    original_first_payment = (
        raw_first_payment.get(loan.loan_id, "")
    )

    audit_rows.append({
        "loan_id": loan.loan_id,
        "vintage": loan.vintage,

        "coverage_flag": loan.coverage_flag,
        "timing_flag": loan.timing_flag,

        "first_payment_saved": loan.first_payment,
        "first_payment_raw": original_first_payment,

        "first_payment_matches_raw":
            loan.first_payment == original_first_payment,

        "rows_saved": loan.monthly_records,
        "rows_raw": len(records),
        "raw_rows_match":
            loan.monthly_records == len(records),

        "first_period_saved": loan.first_period,
        "first_period_raw": first_period_raw,

        "last_period_saved": loan.last_period,
        "last_period_raw": last_period_raw,

        "calendar_span": loan.calendar_span,
        "missing_month_count": len(missing),
        "duplicate_month_count": len(duplicates),

        "first90_saved": first_90_saved,
        "first90_raw": first_90_raw,
        "first90_matches_raw":
            first_90_saved == first_90_raw,

        "months_to_first90_saved":
            loan.months_to_first_90
    })

    # Preserve full monthly histories
    # for the six timing anomalies
    if loan.loan_id in timing_ids:

        for period, dpd, code, date in sorted(records):

            timing_history.append({
                "loan_id": loan.loan_id,
                "vintage": loan.vintage,
                "first_payment_raw":
                    original_first_payment,
                "reporting_period": period,
                "delinquency": dpd,
                "zero_code": code,
                "zero_effective": date
            })


# ==============================================
# 4. Export audit reports
# ==============================================

audit = pd.DataFrame(audit_rows)

audit.to_csv(
    OUTPUTS / "05_flagged_loan_audit.csv",
    index=False
)

pd.DataFrame(
    missing_rows,
    columns=[
        "loan_id",
        "vintage",
        "missing_period"
    ]
).to_csv(
    OUTPUTS / "05_missing_months.csv",
    index=False
)

pd.DataFrame(
    duplicate_rows,
    columns=[
        "loan_id",
        "vintage",
        "duplicate_period",
        "count"
    ]
).to_csv(
    OUTPUTS / "05_duplicate_months.csv",
    index=False
)

pd.DataFrame(timing_history).to_csv(
    OUTPUTS / "05_timing_anomaly_history.csv",
    index=False
)


# ==============================================
# 5. Print concise summary
# ==============================================

print("\n=== Audit Summary ===")

print(f"Flagged loans: {len(audit):,}")
print(f"Missing months: {len(missing_rows):,}")
print(f"Duplicated months: {len(duplicate_rows):,}")

print(
    "Raw row-count mismatches:",
    int((~audit["raw_rows_match"]).sum())
)

print(
    "First 90+ mismatches:",
    int((~audit["first90_matches_raw"]).sum())
)

print(
    "First-payment mismatches:",
    int((~audit["first_payment_matches_raw"]).sum())
)

print("\n=== Timing Anomalies ===")

print(
    audit.loc[
        audit["timing_flag"],
        [
            "loan_id",
            "vintage",
            "first_payment_raw",
            "first_period_raw",
            "first90_raw",
            "months_to_first90_saved",
            "missing_month_count"
        ]
    ].to_string(index=False)
)

print("\n=== First Payment Date vs Vintage ===")

print(
    population.to_string(index=False)
)

print("\nEvent Time Validation completed.")
