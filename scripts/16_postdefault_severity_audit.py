
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/lgd_audit"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_CODES = ["02", "03", "09"]
SALE_CODES = ["15"]

df = pd.read_parquet(DATA / "13_default_to_loss_followup.parquet")
qa = pd.read_parquet(DATA / "12_lgd_definition_audit.parquet")

if df["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan_id in followup file")
if qa["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan_id in LGD audit file")

for frame in (df, qa):
    frame["loan_id"] = frame["loan_id"].astype("string")
    frame["exit_code"] = (
        frame["exit_code"].astype("string").str.strip().str.zfill(2)
    )
    for col in ["actual_loss", "upb_at_first90", "zero_removal_upb"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.loc[~np.isfinite(frame[col]), col] = np.nan

# 1. Reproduce the common 60M cohort
cohort = df.loc[
    df["admitted_60"].fillna(False)
    & df["valid_event_timeline"].fillna(False)
].copy()

exited = cohort.loc[cohort["exit_within_60"].fillna(False)].copy()
main = exited.loc[exited["exit_code"].isin(MAIN_CODES)].copy()
sale = exited.loc[exited["exit_code"].isin(SALE_CODES)].copy()
all_applicable = pd.concat([main, sale], ignore_index=True)

if main.empty:
    raise ValueError("No eligible credit-resolution loans found")

print("\n=== 1. Cohort Funnel ===")
print(f"All first-90 records:     {len(df):,}")
print(f"Common eligible 60M:      {len(cohort):,}")
print(f"All exits within 60M:     {len(exited):,}")
print(f"Credit resolution 02/03/09: {len(main):,}")
print(f"Whole-loan sale 15:       {len(sale):,}")

if len(cohort) != 24324:
    print("WARNING: Cohort count differs from prior audit (24,324).")


# 2. Compute signed LGD without clipping
def prepare(frame):
    frame = frame.copy()

    frame["valid_ead"] = (
        frame["upb_at_first90"].notna()
        & (frame["upb_at_first90"] > 0)
    )
    frame["reported"] = frame["actual_loss"].notna()

    frame["loss_class"] = np.select(
        [
            frame["actual_loss"] > 0,
            frame["actual_loss"] == 0,
            frame["actual_loss"] < 0,
        ],
        ["positive", "zero", "negative"],
        default="missing",
    )

    denominator = frame["upb_at_first90"].where(frame["valid_ead"])
    frame["lgd_signed"] = frame["actual_loss"] / denominator

    frame["valid_exit_upb"] = (
        frame["zero_removal_upb"].notna()
        & (frame["zero_removal_upb"] > 0)
    )
    exit_denominator = frame["zero_removal_upb"].where(
        frame["valid_exit_upb"]
    )
    frame["lgd_exit_upb"] = frame["actual_loss"] / exit_denominator

    return frame


main = prepare(main)
sale = prepare(sale)
all_applicable = prepare(all_applicable)

# 3. Coverage and loss status
print("\n=== 2. Actual Loss Coverage ===")

status = pd.crosstab(
    all_applicable["exit_code"],
    all_applicable["loss_class"],
)

status = status.reindex(
    index=MAIN_CODES + SALE_CODES,
    columns=["positive", "zero", "negative", "missing"],
    fill_value=0,
)
status["total"] = status.sum(axis=1)
print(status.to_string())

print("\n=== 3. Missing / Invalid EAD ===")
print(f"Main exits:             {len(main):,}")
print(f"Missing Actual Loss:    {(~main['reported']).sum():,}")
print(f"Missing/invalid EAD:    {(~main['valid_ead']).sum():,}")
print(
    "Usable LGD observations:",
    main["lgd_signed"].notna().sum(),
)

missing = main.loc[~main["reported"]]
print(
    "Missing loss with defect flag:",
    int(missing["has_defect"].fillna(False).sum()),
)

# 4. Distribution summaries
def summarize(frame, label):
    valid = frame.loc[frame["lgd_signed"].notna()]
    s = valid["lgd_signed"]

    result = {
        "segment": label,
        "exits": len(frame),
        "reported_loss": int(frame["reported"].sum()),
        "valid_ead": int(frame["valid_ead"].sum()),
        "n_lgd": len(s),
        "lgd_coverage_pct": (
            100 * len(s) / len(frame) if len(frame) else np.nan
        ),
        "mean": s.mean(),
        "median": s.median(),
        "p05": s.quantile(0.05),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
        "negative_pct": 100 * (s < 0).mean() if len(s) else np.nan,
        "above_100_pct": 100 * (s > 1).mean() if len(s) else np.nan,
        "weighted_lgd": (
            valid["actual_loss"].sum() / valid["upb_at_first90"].sum()
            if len(valid) else np.nan
        ),
    }
    return result


overall = pd.DataFrame([
    summarize(main, "Credit resolution 02/03/09"),
    summarize(sale, "Whole-loan sale 15"),
    summarize(all_applicable, "All applicable 02/03/09/15"),
])

by_vintage = pd.DataFrame([
    {"vintage": vintage, **summarize(g, str(vintage))}
    for vintage, g in main.groupby("vintage", sort=True)
])

by_code = pd.DataFrame([
    {"exit_code": code, **summarize(g, code)}
    for code, g in all_applicable.groupby("exit_code", sort=True)
])

overall.to_csv(OUT / "severity_overall.csv", index=False)
by_vintage.to_csv(OUT / "severity_by_vintage.csv", index=False)
by_code.to_csv(OUT / "severity_by_exit_code.csv", index=False)
status.to_csv(OUT / "loss_status_by_exit_code.csv")

print("\n=== 4. Overall Signed LGD ===")
cols = [
    "segment", "n_lgd", "mean", "median", "p95",
    "negative_pct", "above_100_pct", "weighted_lgd"
]
print(overall[cols].round(4).to_string(index=False))

print("\n=== 5. By Vintage: Credit Resolution Only ===")
print(
    by_vintage[
        ["vintage", "n_lgd", "mean", "median", "p95"]
    ].round(4).to_string(index=False)
)

print("\n=== 6. By Exit Code ===")
print(
    by_code[
        ["exit_code", "n_lgd", "mean", "median", "p95"]
    ].round(4).to_string(index=False)
)

# 5. Cross-check against Task 12
print("\n=== 7. Cross-file Consistency ===")

audit = main[
    ["loan_id", "actual_loss", "upb_at_first90", "zero_removal_upb"]
].merge(
    qa[
        ["loan_id", "actual_loss", "upb_at_first90", "zero_removal_upb"]
    ],
    on="loan_id",
    how="left",
    suffixes=("_13", "_12"),
    indicator=True,
    validate="one_to_one",
)

print("Loans absent from Task 12:", int((audit["_merge"] == "left_only").sum()))

matched = audit.loc[audit["_merge"] == "both"].copy()
mismatches = []

for field in ["actual_loss", "upb_at_first90", "zero_removal_upb"]:
    a = matched[f"{field}_13"].to_numpy(dtype=float)
    b = matched[f"{field}_12"].to_numpy(dtype=float)

    different = ~np.isclose(
        a, b, rtol=0, atol=0.01, equal_nan=True
    )
    print(f"{field} mismatches (> $0.01): {int(different.sum())}")

    if different.any():
        bad = matched.loc[
            different,
            ["loan_id", f"{field}_13", f"{field}_12"]
        ].copy()
        bad["field"] = field
        bad.columns = ["loan_id", "value_13", "value_12", "field"]
        mismatches.append(bad)

if mismatches:
    pd.concat(mismatches).to_csv(
        OUT / "cross_file_mismatches.csv", index=False
    )

# 6. Diagnostic figure
s = main["lgd_signed"].dropna()
view = s.loc[s.between(-0.5, 1.5)]
outside_pct = 100 * (len(s) - len(view)) / len(s)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(view, bins=75, edgecolor="none")
ax.axvline(0, linestyle="--", linewidth=1, label="Zero loss")
ax.axvline(1, linestyle="--", linewidth=1, label="100% LGD")
ax.set_xlabel("Signed LGD = Actual Loss / First-90 UPB")
ax.set_ylabel("Loan count")
ax.set_title("Post-90 Credit Resolution: Signed LGD Distribution")
ax.legend()
ax.text(
    0.98, 0.95,
    f"Full sample N = {len(s):,}\n"
    f"Outside plotted range: {outside_pct:.2f}%",
    transform=ax.transAxes,
    ha="right", va="top",
)
fig.tight_layout()
fig.savefig(OUT / "signed_lgd_distribution.png", dpi=160)
plt.close(fig)

print("\n=== Audit Complete ===")
print("Outputs:", OUT)
print("Note: Distribution statistics use all valid observations.")
print("The histogram only displays LGD between -50% and 150%.")
print("This is observed post-resolution severity, not predicted LGD.")
