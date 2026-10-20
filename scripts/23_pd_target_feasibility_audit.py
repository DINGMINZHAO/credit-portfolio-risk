
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# Configuration
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/pd_audit"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "03_loan_events_2006_2022.parquet"

CUTOFF = 2026 * 12 + 3
HORIZON = 36

TARGET_CODES = {"02", "03", "09"}
COMPETING_CODES = {"01", "15", "16", "96"}
KNOWN_CODES = TARGET_CODES | COMPETING_CODES

SNAPSHOTS = [
    (2007, 1),
    (2008, 1),
    (2010, 1),
    (2014, 1),
    (2017, 1),
    (2019, 1),
    (2020, 1),
    (2021, 1),
    (2022, 1),
    (2023, 3),
]


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


def normalize_code(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace("", pd.NA)
        .str.zfill(2)
    )


print("\n=== 1. Loading Loan Events ===")

df = pd.read_parquet(
    SOURCE,
    columns=[
        "loan_id",
        "vintage",
        "first_period",
        "last_period",
        "first_90_period",
        "zero_code",
        "zero_effective",
    ],
)

if df["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan identifiers")

df["first_m"] = month_index(df["first_period"])
df["last_m"] = month_index(df["last_period"])
df["first90_m"] = month_index(df["first_90_period"])
df["exit_m"] = month_index(df["zero_effective"])
df["exit_code"] = normalize_code(df["zero_code"])

print(f"Total loans: {len(df):,}")
print(f"Invalid first reporting month: {df['first_m'].isna().sum():,}")
print(f"Invalid last reporting month: {df['last_m'].isna().sum():,}")

invalid_history = (
    df["first_m"].notna()
    & df["last_m"].notna()
    & (df["last_m"] < df["first_m"])
)

print(f"Invalid reporting order: {invalid_history.sum():,}")

if invalid_history.any():
    raise RuntimeError("Invalid reporting-period order")


# ==================================================
# Historical snapshot audit
# ==================================================

results = []

for year, month in SNAPSHOTS:

    snapshot = year * 12 + month
    horizon_end = snapshot + HORIZON

    if horizon_end > CUTOFF:
        raise ValueError("Snapshot lacks full calendar horizon")

    snapshot_name = f"{year}-{month:02d}"

    # Candidate portfolio:
    # Loan reporting interval covers snapshot,
    # and loan has not exited on/before snapshot.
    active = df.loc[
        (df["first_m"] <= snapshot)
        & (df["last_m"] >= snapshot)
        & (
            df["exit_m"].isna()
            | (df["exit_m"] > snapshot)
        )
    ].copy()

    if active.empty:
        continue

    # Exit occurs after snapshot, within 36 months.
    exit_in_window = (
        active["exit_m"].notna()
        & (active["exit_m"] > snapshot)
        & (active["exit_m"] <= horizon_end)
    )

    target = (
        exit_in_window
        & active["exit_code"].isin(TARGET_CODES)
    )

    competing = (
        exit_in_window
        & active["exit_code"].isin(COMPETING_CODES)
    )

    # Flag inconsistent or unrecognised exit metadata.
    code_present = active["exit_code"].notna()
    date_present = active["exit_m"].notna()

    bad_exit_metadata = (
        (code_present != date_present)
        | (
            code_present
            & ~active["exit_code"].isin(KNOWN_CODES)
        )
    )

    # A negative label requires:
    # - confirmed competing exit, OR
    # - observation extending to horizon end,
    #   with no target exit during the horizon.
    observed_to_end = active["last_m"] >= horizon_end

    no_target = (
        observed_to_end
        & ~exit_in_window
        & ~bad_exit_metadata
    )

    outcome = np.select(
        [
            target,
            competing,
            no_target,
        ],
        [
            "target_credit_exit",
            "competing_exit",
            "no_target_observed",
        ],
        default="unknown",
    )

    active["outcome"] = outcome

    counts = active["outcome"].value_counts()

    n_target = int(counts.get("target_credit_exit", 0))
    n_competing = int(counts.get("competing_exit", 0))
    n_no_target = int(counts.get("no_target_observed", 0))
    n_unknown = int(counts.get("unknown", 0))

    n_known = n_target + n_competing + n_no_target

    # Prior first-90 is known at snapshot.
    prior_first90 = (
        active["first90_m"].notna()
        & (active["first90_m"] <= snapshot)
    )

    target_with_prior90 = int(
        (target & prior_first90).sum()
    )

    results.append({
        "snapshot": snapshot_name,
        "portfolio_loans": len(active),
        "target_exits": n_target,
        "competing_exits": n_competing,
        "no_target_observed": n_no_target,
        "unknown": n_unknown,
        "known_labels": n_known,
        "known_label_pct": (
            100 * n_known / len(active)
        ),
        "target_rate_pct": (
            100 * n_target / n_known
            if n_known else np.nan
        ),
        "prior_first90_loans": int(prior_first90.sum()),
        "target_with_prior90": target_with_prior90,
        "bad_exit_metadata": int(
            bad_exit_metadata.sum()
        ),
    })


# ==================================================
# Summary
# ==================================================

summary = pd.DataFrame(results)

if summary.empty:
    raise RuntimeError("No eligible historical snapshots")

print("\n=== 2. Snapshot Feasibility ===")

columns = [
    "snapshot",
    "portfolio_loans",
    "target_exits",
    "competing_exits",
    "no_target_observed",
    "unknown",
    "known_label_pct",
    "target_rate_pct",
]

print(
    summary[columns]
    .round(3)
    .to_string(index=False)
)

print("\n=== 3. Prior Serious Delinquency ===")

print(
    summary[
        [
            "snapshot",
            "prior_first90_loans",
            "target_with_prior90",
        ]
    ].to_string(index=False)
)

print("\n=== 4. Metadata Issues ===")

print(
    summary[
        ["snapshot", "bad_exit_metadata"]
    ].to_string(index=False)
)

# Save summary
output = OUT / "23_snapshot_target_feasibility.csv"
summary.to_csv(output, index=False)

print("\n=== 5. Important Notes ===")
print("Target = credit exit 02/03/09 within 36 months.")
print("Known competing exits are separate from target exits.")
print("Unknown observations are not assigned zero labels.")
print("Reporting intervals do not prove monthly continuity.")
print("Loan balances and point-in-time features not yet audited.")
print("Repeated loans across snapshots are not independent.")

print(f"\nSaved: {output}")
print("\nTask 23 audit complete.")
