
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs/pd_audit"

OUT.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "03_loan_events_2006_2022.parquet"

SNAPSHOTS = [
    (2010, 1),
    (2020, 1),
]

# Validated Performance column positions (zero-based)
COL = {
    "loan_id": 0,
    "period": 1,
    "current_upb": 2,
    "delinquency": 3,
}


# ============================================================
# 2. Helpers
# ============================================================

def month_index(series):
    x = pd.to_numeric(series, errors="coerce")

    year = np.floor(x / 100)
    month = x % 100

    valid = (
        x.notna()
        & (x % 1 == 0)
        & year.between(1990, 2030)
        & month.between(1, 12)
    )

    return (year * 12 + month).where(valid)


def parse_float(value):
    try:
        result = float(value.strip())
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# 3. Load loan-level event table
# ============================================================

print("\n=== 1. Loading Loan Events ===")

df = pd.read_parquet(
    SOURCE,
    columns=[
        "loan_id",
        "vintage",
        "first_period",
        "last_period",
        "zero_effective",
    ],
)

df["loan_id"] = df["loan_id"].astype("string")

if df["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan IDs")

df["first_m"] = month_index(df["first_period"])
df["last_m"] = month_index(df["last_period"])
df["exit_m"] = month_index(df["zero_effective"])

print(f"Loaded loans: {len(df):,}")


# ============================================================
# 4. Reconstruct candidate snapshots
# ============================================================

print("\n=== 2. Building Snapshot Candidates ===")

frames = []

for year, month in SNAPSHOTS:

    snapshot_yyyymm = year * 100 + month
    snapshot_m = year * 12 + month

    active = df.loc[
        (df["first_m"] <= snapshot_m)
        & (df["last_m"] >= snapshot_m)
        & (
            df["exit_m"].isna()
            | (df["exit_m"] > snapshot_m)
        )
    ].copy()

    active["snapshot"] = snapshot_yyyymm

    frames.append(
        active[
            ["loan_id", "vintage", "snapshot"]
        ].copy()
    )

    print(
        f"{snapshot_yyyymm}: "
        f"candidate loans = {len(active):,}"
    )

candidates = pd.concat(
    frames,
    ignore_index=True,
)

if candidates.duplicated(
    ["loan_id", "snapshot"]
).any():
    raise ValueError("Duplicate loan-snapshot pairs")


# ============================================================
# 5. Extract exact monthly Performance records
# ============================================================

print("\n=== 3. Scanning Performance ZIPs ===")

found_records = []

for vintage in sorted(
    candidates["vintage"].unique()
):

    selected = candidates.loc[
        candidates["vintage"] == vintage
    ]

    if selected.empty:
        continue

    # Loan ID -> requested snapshot months
    wanted = {}

    for r in selected.itertuples(index=False):
        wanted.setdefault(
            str(r.loan_id), set()
        ).add(int(r.snapshot))

    zip_path = RAW / f"sample_{vintage}.zip"
    member = f"sample_perf_{vintage}.txt"

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    hits = {}

    with zipfile.ZipFile(zip_path) as archive:

        with archive.open(member) as f:

            for line_number, raw in enumerate(f, 1):

                # Read ID and period first.
                # Avoid parsing all 35 fields unnecessarily.
                head = raw.split(b"|", 2)

                if len(head) < 3:
                    raise ValueError(
                        f"Malformed row: {vintage}, {line_number}"
                    )

                loan_id = head[0].decode(
                    "utf-8", errors="replace"
                ).strip()

                desired = wanted.get(loan_id)

                if desired is None:
                    continue

                period_bytes = head[1].strip()

                if not period_bytes.isdigit():
                    continue

                period = int(period_bytes)

                if period not in desired:
                    continue

                parts = raw.rstrip(
                    b"\r\n"
                ).split(b"|")

                if len(parts) != 35:
                    raise ValueError(
                        f"{vintage}, row {line_number}: "
                        f"expected 35 fields, got {len(parts)}"
                    )

                key = (loan_id, period)

                if key not in hits:

                    hits[key] = {
                        "loan_id": loan_id,
                        "snapshot": period,
                        "matches": 0,
                        "current_upb": parse_float(
                            parts[COL["current_upb"]]
                        ),
                        "dpd_raw": parts[
                            COL["delinquency"]
                        ].decode(
                            "utf-8", errors="replace"
                        ).strip(),
                    }

                hits[key]["matches"] += 1

    found_records.extend(hits.values())

    duplicates = sum(
        rec["matches"] > 1
        for rec in hits.values()
    )

    print(
        f"{vintage}: "
        f"requested={len(selected):,}, "
        f"matched={len(hits):,}, "
        f"duplicate_months={duplicates:,}"
    )


# ============================================================
# 6. Join extracted features to candidate portfolio
# ============================================================

print("\n=== 4. Snapshot Coverage ===")

found = pd.DataFrame(
    found_records,
    columns=[
        "loan_id",
        "snapshot",
        "matches",
        "current_upb",
        "dpd_raw",
    ],
)

result = candidates.merge(
    found,
    on=["loan_id", "snapshot"],
    how="left",
    validate="one_to_one",
)

result["matches"] = (
    result["matches"]
    .fillna(0)
    .astype(int)
)

result["has_exact_record"] = (
    result["matches"] == 1
)

result["valid_upb"] = (
    result["has_exact_record"]
    & np.isfinite(result["current_upb"])
    & (result["current_upb"] > 0)
)

result["valid_dpd"] = (
    result["has_exact_record"]
    & result["dpd_raw"].notna()
    & result["dpd_raw"].astype("string").str.len().gt(0)
).fillna(False)

summary = result.groupby("snapshot").agg(
    candidates=("loan_id", "size"),
    exact_records=("has_exact_record", "sum"),
    missing_records=(
        "matches",
        lambda x: int((x == 0).sum()),
    ),
    duplicate_records=(
        "matches",
        lambda x: int((x > 1).sum()),
    ),
    valid_upb=("valid_upb", "sum"),
    valid_dpd=("valid_dpd", "sum"),
)

summary["exact_coverage_pct"] = (
    100
    * summary["exact_records"]
    / summary["candidates"]
)

print(summary.round(3).to_string())


# ============================================================
# 7. Inspect DPD values
# ============================================================

print("\n=== 5. Observed DPD Codes ===")

for snapshot, group in result.groupby("snapshot"):

    print(f"\nSnapshot {snapshot}")

    print(
        group.loc[
            group["has_exact_record"],
            "dpd_raw"
        ]
        .value_counts(dropna=False)
        .head(12)
        .to_string()
    )


# ============================================================
# 8. Save audit results
# ============================================================

result.to_parquet(
    DATA / "24_snapshot_feature_probe.parquet",
    index=False,
)

summary.to_csv(
    OUT / "24_snapshot_coverage.csv"
)

issues = result.loc[
    ~result["has_exact_record"]
    | ~result["valid_upb"]
    | ~result["valid_dpd"]
].copy()

issues.to_csv(
    OUT / "24_snapshot_issues.csv",
    index=False,
)

print("\n=== 6. Final Summary ===")

print(f"Loan-snapshot pairs: {len(result):,}")

print(
    "Missing exact records:",
    f"{int((result['matches'] == 0).sum()):,}"
)

print(
    "Duplicate exact records:",
    f"{int((result['matches'] > 1).sum()):,}"
)

print(
    "Invalid current UPB:",
    f"{int((~result['valid_upb']).sum()):,}"
)

print(
    "Invalid/missing DPD:",
    f"{int((~result['valid_dpd']).sum()):,}"
)

print("\nTask 24 snapshot audit complete.")
