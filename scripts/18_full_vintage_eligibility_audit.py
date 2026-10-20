
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/processed/03_loan_events_2006_2022.parquet"

df = pd.read_parquet(
    PATH,
    columns=[
        "vintage",
        "first_90_period",
        "zero_code",
        "zero_effective",
    ],
)

# Convert YYYYMM dates to a continuous month index
def month_index(series):
    x = pd.to_numeric(series, errors="coerce")
    year = x // 100
    month = x % 100

    valid = (
        x.notna()
        & (x % 1 == 0)
        & year.between(1990, 2030)
        & month.between(1, 12)
    )

    return (year * 12 + month).where(valid)

df["first90_m"] = month_index(df["first_90_period"])
df["exit_m"] = month_index(df["zero_effective"])

df["exit_code"] = (
    df["zero_code"]
    .astype("string")
    .str.strip()
    .str.zfill(2)
)

# Release 47 observation cutoff: March 2026
cutoff = 2026 * 12 + 3

df["first90"] = df["first90_m"].notna()

# Calendar maturity only; not final eligibility
df["mature_36"] = (
    df["first90"] & (df["first90_m"] + 36 <= cutoff)
)
df["mature_60"] = (
    df["first90"] & (df["first90_m"] + 60 <= cutoff)
)

df["months_to_exit"] = df["exit_m"] - df["first90_m"]

df["credit_exit_60"] = (
    df["first90"]
    & df["exit_code"].isin(["02", "03", "09"])
    & df["months_to_exit"].between(0, 60)
)

df["credit_exit_60_mature"] = (
    df["credit_exit_60"] & df["mature_60"]
)

summary = df.groupby("vintage").agg(
    loans=("vintage", "size"),
    first90=("first90", "sum"),
    mature36=("mature_36", "sum"),
    mature60=("mature_60", "sum"),
    credit_exit60=("credit_exit_60", "sum"),
    credit_exit60_mature=(
        "credit_exit_60_mature", "sum"
    ),
)

summary = summary.astype(int)

print("\n=== FULL VINTAGE FEASIBILITY ===")
print(summary.to_string())

print("\n=== TOTAL ===")
print(summary.sum().to_string())

print("\nNotes:")
print("mature36/60 = calendar maturity only.")
print("credit_exit60 = observed credit exits within 60M.")
print("Actual Loss and UPB coverage not yet checked.")
print("Final eligibility requires timeline validation.")
