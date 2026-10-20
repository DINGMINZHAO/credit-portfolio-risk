
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/lgd_audit"
OUT.mkdir(parents=True, exist_ok=True)

CODES = ["02", "03", "09", "15"]
MAIN = ["02", "03", "09"]


def normalize_code(s):
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def month_index(s):
    x = pd.to_numeric(s, errors="coerce")
    year = np.floor(x / 100)
    month = x % 100

    valid = (
        np.isfinite(x)
        & (x % 1 == 0)
        & year.between(1990, 2030)
        & month.between(1, 12)
    )
    return (year * 12 + month).where(valid)


def numeric_equal(a, b):
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)
    return np.isclose(x, y, rtol=0, atol=0.01, equal_nan=True)


print("\n=== 1. Loading Audited Data ===")

old = pd.read_parquet(
    DATA / "12_lgd_definition_audit.parquet",
    columns=[
        "loan_id", "vintage", "exit_code",
        "first90_recomputed", "exit_date",
        "actual_loss", "upb_at_first90",
    ],
)

new = pd.read_parquet(
    DATA / "19_post90_exited_lgd_2012_2022.parquet",
    columns=[
        "loan_id", "vintage", "exit_code",
        "first90_recomputed", "raw_exit_date",
        "expected_first90", "expected_exit",
        "raw_exit_code",
        "actual_loss", "upb_at_first90",
    ],
)

# Verify newly extracted dates and exit codes
first90_ok = numeric_equal(
    new["first90_recomputed"], new["expected_first90"]
)
exit_date_ok = numeric_equal(
    new["raw_exit_date"], new["expected_exit"]
)
exit_code_ok = (
    normalize_code(new["raw_exit_code"])
    == normalize_code(new["exit_code"])
).fillna(False)

print("New first-90 mismatches:", int((~first90_ok).sum()))
print("New exit-date mismatches:", int((~exit_date_ok).sum()))
print("New exit-code mismatches:", int((~exit_code_ok).sum()))

if not (first90_ok.all() and exit_date_ok.all() and exit_code_ok.all()):
    raise RuntimeError("New extraction validation failed")

old = old.rename(columns={"exit_date": "exit_period"})
new = new.rename(columns={"raw_exit_date": "exit_period"})

common = [
    "loan_id", "vintage", "exit_code",
    "first90_recomputed", "exit_period",
    "actual_loss", "upb_at_first90",
]

df = pd.concat(
    [old[common], new[common]],
    ignore_index=True,
)

if df["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan identifiers")

df["exit_code"] = normalize_code(df["exit_code"])
df["first90_m"] = month_index(df["first90_recomputed"])
df["exit_m"] = month_index(df["exit_period"])

print("Loaded records:", f"{len(df):,}")


print("\n=== 2. Consistent Eligibility ===")

df["valid_timeline"] = (
    df["first90_m"].notna()
    & df["exit_m"].notna()
    & (df["exit_m"] >= df["first90_m"])
)

eligible = df.loc[
    df["exit_code"].isin(CODES)
    & df["valid_timeline"]
].copy()

print("Valid event-to-exit timeline:", f"{len(eligible):,}")
print("Excluded by timeline/code:",
      f"{len(df) - len(eligible):,}")


print("\n=== 3. LGD Calculation ===")

eligible["actual_loss"] = pd.to_numeric(
    eligible["actual_loss"], errors="coerce"
)
eligible["upb_at_first90"] = pd.to_numeric(
    eligible["upb_at_first90"], errors="coerce"
)

eligible["valid_ead"] = (
    np.isfinite(eligible["upb_at_first90"])
    & (eligible["upb_at_first90"] > 0)
)

eligible["reported_loss"] = np.isfinite(
    eligible["actual_loss"]
)

eligible["valid_lgd"] = (
    eligible["valid_ead"]
    & eligible["reported_loss"]
)

eligible["lgd_signed"] = np.where(
    eligible["valid_lgd"],
    eligible["actual_loss"] / eligible["upb_at_first90"],
    np.nan,
)

eligible["months_to_exit"] = (
    eligible["exit_m"] - eligible["first90_m"]
)

# Do not clip LGD or replace missing loss with zero.


def summarize(g):
    v = g.loc[g["valid_lgd"]]
    s = v["lgd_signed"]

    return pd.Series({
        "exits": len(g),
        "valid_lgd": len(v),
        "coverage_pct": (
            100 * len(v) / len(g) if len(g) else np.nan
        ),
        "mean_lgd": s.mean(),
        "median_lgd": s.median(),
        "p95_lgd": s.quantile(0.95),
        "weighted_lgd": (
            v["actual_loss"].sum() / v["upb_at_first90"].sum()
            if len(v) else np.nan
        ),
        "negative_lgd": int((s < 0).sum()),
        "above_100_lgd": int((s > 1).sum()),
        "median_months_to_exit": (
            v["months_to_exit"].median()
        ),
    })


main = eligible.loc[
    eligible["exit_code"].isin(MAIN)
].copy()

sale = eligible.loc[
    eligible["exit_code"] == "15"
].copy()

by_vintage = (
    main.groupby("vintage")
    .apply(summarize, include_groups=False)
    .reset_index()
)

by_code = (
    eligible.groupby("exit_code")
    .apply(summarize, include_groups=False)
    .reset_index()
)

print("\n=== 4. Main Credit Resolution by Vintage ===")

cols = [
    "vintage", "exits", "valid_lgd",
    "mean_lgd", "median_lgd", "weighted_lgd"
]
print(by_vintage[cols].round(4).to_string(index=False))

print("\n=== 5. By Exit Code ===")

cols = [
    "exit_code", "exits", "valid_lgd",
    "mean_lgd", "median_lgd", "weighted_lgd"
]
print(by_code[cols].round(4).to_string(index=False))

print("\n=== 6. Overall ===")

for name, group in [
    ("Credit resolution 02/03/09", main),
    ("Whole-loan sale 15", sale),
]:
    stats = summarize(group)
    print(f"\n{name}")
    print(stats.round(4).to_string())


print("\n=== 7. Saving Results ===")

eligible.to_parquet(
    DATA / "20_unified_lgd_2006_2022.parquet",
    index=False,
)

by_vintage.to_csv(
    OUT / "20_lgd_by_vintage.csv", index=False
)

by_code.to_csv(
    OUT / "20_lgd_by_exit_code.csv", index=False
)

print("Saved unified LGD dataset and summary tables.")
print("This is a historical completed-exit sample.")
print("No 60-month eligibility restriction was applied.")
