from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "lgd_audit"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_CODES = ["02", "03", "09"]
ALL_CODES = ["02", "03", "09", "15"]

def normalize_code(s):
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

# -----------------------------
# 1. Load unified dataset
# -----------------------------
df = pd.read_parquet(DATA / "20_unified_lgd_2006_2022.parquet")

df["exit_code"] = normalize_code(df["exit_code"])

# -----------------------------
# 2. Reconstruct exclusion reasons
# -----------------------------
full = df.copy()

full["reason"] = "included"

full.loc[~full["exit_code"].isin(ALL_CODES), "reason"] = "invalid_exit_code"
full.loc[full["first90_m"].isna(), "reason"] = "missing_first90"
full.loc[full["exit_m"].isna(), "reason"] = "missing_exit_date"
full.loc[
    full["first90_m"].notna()
    & full["exit_m"].notna()
    & (full["exit_m"] < full["first90_m"]),
    "reason"
] = "exit_before_first90"

exclusion = (
    full["reason"]
    .value_counts(dropna=False)
    .rename_axis("reason")
    .reset_index(name="loans")
)

excluded = full.loc[full["reason"] != "included"].copy()

# -----------------------------
# 3. Main LGD sample
# -----------------------------
main = df.loc[
    df["exit_code"].isin(MAIN_CODES)
    & df["valid_lgd"].fillna(False)
].copy()

sale = df.loc[
    (df["exit_code"] == "15")
    & df["valid_lgd"].fillna(False)
].copy()

s = main["lgd_signed"].dropna().astype(float)

if s.empty:
    raise RuntimeError("No valid main LGD observations found")

# -----------------------------
# 4. Summary statistics
# -----------------------------
summary = pd.Series({
    "n_main_valid_lgd": len(s),
    "mean": s.mean(),
    "std": s.std(),
    "min": s.min(),
    "p01": s.quantile(0.01),
    "p05": s.quantile(0.05),
    "p10": s.quantile(0.10),
    "p25": s.quantile(0.25),
    "median": s.quantile(0.50),
    "p75": s.quantile(0.75),
    "p90": s.quantile(0.90),
    "p95": s.quantile(0.95),
    "p99": s.quantile(0.99),
    "max": s.max(),
    "negative_count": int((s < 0).sum()),
    "negative_pct": 100 * (s < 0).mean(),
    "above_100_count": int((s > 1).sum()),
    "above_100_pct": 100 * (s > 1).mean(),
    "weighted_lgd": (
        main["actual_loss"].sum() / main["upb_at_first90"].sum()
    ),
    "median_months_to_exit": main["months_to_exit"].median(),
})

summary.to_csv(OUT / "21_lgd_overall_summary.csv", header=["value"])

# -----------------------------
# 5. Extreme-value diagnostic
# -----------------------------
extreme = main.loc[
    (main["lgd_signed"] < 0) | (main["lgd_signed"] > 1)
].copy()

extreme = extreme[
    [
        "loan_id",
        "vintage",
        "exit_code",
        "actual_loss",
        "upb_at_first90",
        "lgd_signed",
        "months_to_exit",
    ]
].sort_values(
    ["lgd_signed", "actual_loss"],
    ascending=[False, False]
)

extreme.to_csv(OUT / "21_lgd_extreme_cases.csv", index=False)
exclusion.to_csv(OUT / "21_lgd_exclusion_reasons.csv", index=False)

# -----------------------------
# 6. Figure 1: Histogram
# -----------------------------
plot_range = (-0.5, 1.5)
view = s[s.between(plot_range[0], plot_range[1])]
outside_pct = 100 * (len(s) - len(view)) / len(s)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(view, bins=80)
ax.axvline(0, linestyle="--", linewidth=1, label="LGD = 0")
ax.axvline(1, linestyle="--", linewidth=1, label="LGD = 100%")
ax.axvline(summary["median"], linestyle="--", linewidth=1, label=f"Median = {summary['median']:.2f}")
ax.set_xlabel("LGD = Actual Loss / UPB at First 90+")
ax.set_ylabel("Loan Count")
ax.set_title("Overall LGD Distribution (Credit Resolution 02/03/09)")
ax.legend()
ax.text(
    0.98,
    0.95,
    (
        f"N = {len(s):,}\n"
        f"Mean = {summary['mean']:.3f}\n"
        f"Median = {summary['median']:.3f}\n"
        f"P95 = {summary['p95']:.3f}\n"
        f"Outside plotted range = {outside_pct:.2f}%"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
)
fig.tight_layout()
fig.savefig(OUT / "21_lgd_overall_hist.png", dpi=180)
plt.close(fig)

# -----------------------------
# 7. Figure 2: Empirical CDF
# -----------------------------
x = np.sort(s.to_numpy())
y = np.arange(1, len(x) + 1) / len(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.axvline(summary["median"], linestyle="--", linewidth=1, label=f"Median = {summary['median']:.2f}")
ax.axvline(summary["p95"], linestyle="--", linewidth=1, label=f"P95 = {summary['p95']:.2f}")
ax.axhline(0.50, linestyle="--", linewidth=1)
ax.axhline(0.95, linestyle="--", linewidth=1)
ax.set_xlabel("LGD = Actual Loss / UPB at First 90+")
ax.set_ylabel("Empirical Cumulative Probability")
ax.set_title("Empirical CDF of LGD (Credit Resolution 02/03/09)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "21_lgd_overall_cdf.png", dpi=180)
plt.close(fig)

# -----------------------------
# 8. Console output
# -----------------------------
print("\n=== 1. Exclusion Reasons ===")
print(exclusion.to_string(index=False))

print("\n=== 2. Overall LGD Summary (02/03/09) ===")
print(summary.round(4).to_string())

print("\n=== 3. Whole-loan Sale Reference (15) ===")
if len(sale) > 0:
    sale_stats = pd.Series({
        "n_valid_lgd": len(sale),
        "mean": sale["lgd_signed"].mean(),
        "median": sale["lgd_signed"].median(),
        "p95": sale["lgd_signed"].quantile(0.95),
        "weighted_lgd": sale["actual_loss"].sum() / sale["upb_at_first90"].sum(),
    })
    print(sale_stats.round(4).to_string())
else:
    print("No valid Code 15 sample")

print("\n=== 4. Saved Files ===")
print(OUT / "21_lgd_overall_hist.png")
print(OUT / "21_lgd_overall_cdf.png")
print(OUT / "21_lgd_overall_summary.csv")
print(OUT / "21_lgd_exclusion_reasons.csv")
print(OUT / "21_lgd_extreme_cases.csv")

print("\nAudit complete.")