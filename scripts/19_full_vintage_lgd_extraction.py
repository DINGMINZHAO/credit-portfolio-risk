
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/lgd_audit"

OUT.mkdir(parents=True, exist_ok=True)

# Freddie Mac Release 47 Performance column indices (0-based)
COL = {
    "loan_id": 0,
    "period": 1,
    "current_upb": 2,
    "delinquency": 3,
    "defect_date": 6,
    "exit_code": 8,
    "exit_date": 9,
    "actual_loss": 21,
    "exit_upb": 26,
}

MAIN_CODES = {"02", "03", "09"}
SALE_CODES = {"15"}
ALL_CODES = MAIN_CODES | SALE_CODES

YEARS = list(range(2012, 2023))


# ============================================================
# 2. Parsing helpers
# ============================================================

def number(value):
    value = str(value).strip()

    if not value:
        return np.nan

    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except ValueError:
        return np.nan


def month(value):
    value = str(value).strip()

    if len(value) != 6 or not value.isdigit():
        return None

    year = int(value[:4])
    mon = int(value[4:])

    if not (1990 <= year <= 2030 and 1 <= mon <= 12):
        return None

    return int(value)


def code(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    if value.endswith(".0"):
        value = value[:-2]

    return value.zfill(2)


# ============================================================
# 3. Load historical loan events
# ============================================================

print("\n=== Loading Historical Loan Events ===")

event_path = DATA / "03_loan_events_2006_2022.parquet"

events = pd.read_parquet(
    event_path,
    columns=[
        "loan_id",
        "vintage",
        "first_90_period",
        "zero_code",
        "zero_effective",
    ],
)

events["loan_id"] = events["loan_id"].astype("string")
events["zero_code"] = events["zero_code"].map(code)

events = events.loc[
    events["first_90_period"].notna()
    & events["zero_code"].isin(ALL_CODES)
].copy()

if events["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan IDs detected")

print(f"Selected loan events: {len(events):,}")


# ============================================================
# 4. Extract selected loans from raw Performance ZIP
# ============================================================

def extract_year(year):

    selected = events.loc[
        events["vintage"] == year
    ].copy()

    records = {}

    for r in selected.itertuples(index=False):

        loan_id = str(r.loan_id)

        records[loan_id] = {
            "loan_id": loan_id,
            "vintage": year,
            "exit_code": r.zero_code,
            "expected_first90": int(r.first_90_period),
            "expected_exit": r.zero_effective,
            "first90_recomputed": np.nan,
            "upb_at_first90": np.nan,
            "actual_loss": np.nan,
            "zero_removal_upb": np.nan,
            "raw_exit_code": "",
            "raw_exit_date": np.nan,
            "defect_settlement": np.nan,
            "last_period": np.nan,
            "monthly_rows": 0,
        }

    zip_path = RAW / f"sample_{year}.zip"
    member = f"sample_perf_{year}.txt"

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as archive:

        if member not in archive.namelist():
            raise FileNotFoundError(
                f"{member} missing from {zip_path}"
            )

        with archive.open(member) as f:

            for line_number, raw_line in enumerate(f, 1):

                parts = raw_line.decode(
                    "utf-8-sig",
                    errors="replace",
                ).rstrip("\r\n").split("|")

                if len(parts) != 35:
                    raise ValueError(
                        f"{year}, row {line_number}: "
                        f"expected 35 columns, got {len(parts)}"
                    )

                loan_id = parts[COL["loan_id"]].strip()

                rec = records.get(loan_id)

                if rec is None:
                    continue

                rec["monthly_rows"] += 1

                period = month(parts[COL["period"]])

                if period is None:
                    continue

                # --------------------------------------------
                # First observed 90+ DPD and corresponding UPB
                # --------------------------------------------

                status = parts[COL["delinquency"]].strip()

                if status.isdigit() and int(status) >= 3:

                    old_period = rec["first90_recomputed"]

                    if (
                        pd.isna(old_period)
                        or period < old_period
                    ):
                        rec["first90_recomputed"] = period

                        rec["upb_at_first90"] = number(
                            parts[COL["current_upb"]]
                        )

                # --------------------------------------------
                # Latest monthly record
                # --------------------------------------------

                last_period = rec["last_period"]

                if (
                    pd.isna(last_period)
                    or period >= last_period
                ):

                    rec["last_period"] = period

                    rec["actual_loss"] = number(
                        parts[COL["actual_loss"]]
                    )

                    rec["zero_removal_upb"] = number(
                        parts[COL["exit_upb"]]
                    )

                    rec["raw_exit_code"] = code(
                        parts[COL["exit_code"]]
                    )

                    rec["raw_exit_date"] = number(
                        parts[COL["exit_date"]]
                    )

                    rec["defect_settlement"] = number(
                        parts[COL["defect_date"]]
                    )

    # Convert dict_values into a list for pandas compatibility
    return pd.DataFrame.from_records(
        list(records.values())
    )


# ============================================================
# 5. Validate new parser against audited 2006 data
# ============================================================

def validate_2006(extracted):

    old_path = DATA / "12_lgd_definition_audit.parquet"

    old = pd.read_parquet(
        old_path,
        columns=[
            "loan_id",
            "first90_recomputed",
            "upb_at_first90",
            "actual_loss",
            "zero_removal_upb",
        ],
    )

    old["loan_id"] = old["loan_id"].astype("string")

    old = old.loc[
        old["loan_id"].isin(extracted["loan_id"])
    ].copy()

    merged = extracted.merge(
        old,
        on="loan_id",
        how="left",
        suffixes=("_new", "_old"),
        indicator=True,
        validate="one_to_one",
    )

    missing = int(
        (merged["_merge"] != "both").sum()
    )

    print(f"Unmatched 2006 loans: {missing}")

    fields = [
        "first90_recomputed",
        "upb_at_first90",
        "actual_loss",
        "zero_removal_upb",
    ]

    total_mismatches = 0

    for field in fields:

        a = pd.to_numeric(
            merged[f"{field}_new"],
            errors="coerce",
        ).to_numpy(dtype=float)

        b = pd.to_numeric(
            merged[f"{field}_old"],
            errors="coerce",
        ).to_numpy(dtype=float)

        different = ~np.isclose(
            a,
            b,
            rtol=0,
            atol=0.01,
            equal_nan=True,
        )

        n = int(different.sum())
        total_mismatches += n

        print(f"{field}: {n} mismatches")

        if n > 0:
            examples = merged.loc[
                different,
                [
                    "loan_id",
                    f"{field}_new",
                    f"{field}_old",
                ],
            ].head(5)

            print(examples.to_string(index=False))

    if missing > 0 or total_mismatches > 0:
        raise RuntimeError(
            "2006 parser validation FAILED. "
            "Check differences before continuing."
        )

    print("\n2006 parser validation PASSED")


# ============================================================
# 6. Run baseline validation
# ============================================================

print("\n=== 1. PARSER VALIDATION: 2006 ===")

baseline = extract_year(2006)

validate_2006(baseline)


# ============================================================
# 7. Extract 2012-2022
# ============================================================

print("\n=== 2. EXTRACTING 2012-2022 ===")

frames = []

for year in YEARS:

    extracted = extract_year(year)

    if extracted.empty:
        print(f"{year}: no selected loans")
        continue

    found = extracted["monthly_rows"] > 0

    first90_match = (
        extracted["first90_recomputed"]
        == extracted["expected_first90"]
    ).fillna(False)

    exit_code_match = (
        extracted["raw_exit_code"]
        == extracted["exit_code"]
    ).fillna(False)

    first90_mismatch = int(
        (found & ~first90_match).sum()
    )

    exit_code_mismatch = int(
        (found & ~exit_code_match).sum()
    )

    print(
        f"{year}: "
        f"selected={len(extracted):,}, "
        f"found={int(found.sum()):,}, "
        f"first90_mismatch={first90_mismatch}, "
        f"exit_code_mismatch={exit_code_mismatch}"
    )

    frames.append(extracted)

if not frames:
    raise RuntimeError("No records extracted")

result = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# 8. Calculate LGD
# ============================================================

result["valid_ead"] = (
    result["upb_at_first90"].notna()
    & (result["upb_at_first90"] > 0)
)

result["reported_loss"] = (
    result["actual_loss"].notna()
)

result["lgd_signed"] = (
    result["actual_loss"]
    / result["upb_at_first90"].where(
        result["valid_ead"]
    )
)

result["valid_lgd"] = np.isfinite(
    result["lgd_signed"]
)

result["loss_class"] = np.select(
    [
        result["actual_loss"] > 0,
        result["actual_loss"] == 0,
        result["actual_loss"] < 0,
    ],
    [
        "positive",
        "zero",
        "negative",
    ],
    default="missing",
)

# Keep unresolved records for further audits.
# Do not replace missing loss with zero.
# Do not clip LGD to [0, 1].


# ============================================================
# 9. Save extracted data
# ============================================================

output_file = (
    DATA / "19_post90_exited_lgd_2012_2022.parquet"
)

result.to_parquet(
    output_file,
    index=False,
)

summary = result.groupby(
    ["vintage", "exit_code"]
).agg(
    selected=("loan_id", "size"),
    reported_loss=("reported_loss", "sum"),
    valid_ead=("valid_ead", "sum"),
    valid_lgd=("valid_lgd", "sum"),
).reset_index()

summary.to_csv(
    OUT / "lgd_extension_2012_2022_coverage.csv",
    index=False,
)


# ============================================================
# 10. Output concise results
# ============================================================

print("\n=== 3. COVERAGE BY VINTAGE ===")

year_summary = summary.groupby("vintage")[
    [
        "selected",
        "reported_loss",
        "valid_ead",
        "valid_lgd",
    ]
].sum()

print(year_summary.to_string())

print("\n=== 4. LOSS STATUS ===")

print(
    result["loss_class"]
    .value_counts()
    .to_string()
)

print("\n=== 5. TOTAL ===")

print(f"Selected exits: {len(result):,}")

print(
    "Reported Actual Loss:",
    f"{int(result['reported_loss'].sum()):,}",
)

print(
    "Valid EAD:",
    f"{int(result['valid_ead'].sum()):,}",
)

print(
    "Valid LGD:",
    f"{int(result['valid_lgd'].sum()):,}",
)

print(f"\nSaved: {output_file}")

print("\nExtraction complete.")
print("Note: These are observed exit samples.")
print("Final LGD modeling eligibility is not yet determined.")
