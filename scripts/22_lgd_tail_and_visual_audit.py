
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/lgd_audit"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_CODES = ["02", "03", "09"]
ALL_CODES = MAIN_CODES + ["15"]

PLOT_MIN = -0.5
PLOT_MAX = 1.5


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
        x.notna()
        & (x % 1 == 0)
        & year.between(1990, 2030)
        & month.between(1, 12)
    )

    return (year * 12 + month).where(valid)


# ============================================================
# 2. Reconstruct excluded loans
# ============================================================

old = pd.read_parquet(
    DATA / "12_lgd_definition_audit.parquet",
    columns=[
        "loan_id",
        "vintage",
        "exit_code",
        "first90_recomputed",
        "exit_date",
    ],
).rename(columns={"exit_date": "exit_period"})

new = pd.read_parquet(
    DATA / "19_post90_exited_lgd_2012_2022.parquet",
    columns=[
        "loan_id",
        "vintage",
        "exit_code",
        "first90_recomputed",
        "raw_exit_date",
    ],
).rename(columns={"raw_exit_date": "exit_period"})

raw = pd.concat([old, new], ignore_index=True)

raw["loan_id"] = raw["loan_id"].astype("string")
raw["exit_code"] = normalize_code(raw["exit_code"])

if raw["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan IDs in candidate data")

raw["first90_m"] = month_index(raw["first90_recomputed"])
raw["exit_m"] = month_index(raw["exit_period"])

raw["exclusion_reason"] = np.select(
    [
        ~raw["exit_code"].isin(ALL_CODES),
        raw["first90_m"].isna(),
        raw["exit_m"].isna(),
        (
            raw["first90_m"].notna()
            & raw["exit_m"].notna()
            & (raw["exit_m"] < raw["first90_m"])
        ),
    ],
    [
        "invalid_exit_code",
        "missing_or_invalid_first90",
        "missing_or_invalid_exit_date",
        "exit_before_first90",
    ],
    default="included",
)

excluded = raw.loc[
    raw["exclusion_reason"] != "included"
].copy()

excluded.to_csv(
    OUT / "22_excluded_loans.csv",
    index=False,
)

print("\n=== 1. Exclusion Audit ===")
print(f"Original candidates: {len(raw):,}")
print(raw["exclusion_reason"].value_counts().to_string())

print("\nExcluded by Vintage:")
if not excluded.empty:
    print(
        pd.crosstab(
            excluded["vintage"],
            excluded["exclusion_reason"],
        ).to_string()
    )


# ============================================================
# 3. Verify unified dataset consistency
# ============================================================

df = pd.read_parquet(
    DATA / "20_unified_lgd_2006_2022.parquet"
)

df["loan_id"] = df["loan_id"].astype("string")
df["exit_code"] = normalize_code(df["exit_code"])

if df["loan_id"].duplicated().any():
    raise ValueError("Duplicate loan IDs in unified dataset")

expected_ids = set(
    raw.loc[
        raw["exclusion_reason"] == "included",
        "loan_id",
    ]
)

actual_ids = set(df["loan_id"])

print("\n=== 2. Dataset Consistency ===")
print(f"Expected: {len(expected_ids):,}")
print(f"Actual:   {len(actual_ids):,}")
print(f"Missing:  {len(expected_ids - actual_ids):,}")
print(f"Extra:    {len(actual_ids - expected_ids):,}")

if expected_ids != actual_ids:
    raise RuntimeError("Unified dataset consistency FAILED")

print("Consistency check PASSED")


# ============================================================
# 4. Select valid credit-resolution LGD
# ============================================================

main = df.loc[
    df["exit_code"].isin(MAIN_CODES)
    & df["valid_lgd"].fillna(False)
].copy()

for col in [
    "lgd_signed",
    "actual_loss",
    "upb_at_first90",
]:
    main[col] = pd.to_numeric(
        main[col], errors="coerce"
    )

valid = (
    np.isfinite(main["lgd_signed"])
    & np.isfinite(main["actual_loss"])
    & np.isfinite(main["upb_at_first90"])
    & (main["upb_at_first90"] > 0)
)

main = main.loc[valid].copy()

if main.empty:
    raise RuntimeError("No valid LGD observations")

s = main["lgd_signed"]


# ============================================================
# 5. Tail statistics
# ============================================================

print("\n=== 3. LGD Tail Diagnostics ===")
print(f"Valid loans: {len(main):,}")

thresholds = [
    ("LGD < 0", s < 0),
    ("LGD > 100%", s > 1),
    ("LGD > 150%", s > 1.5),
    ("LGD > 200%", s > 2),
    ("LGD > 500%", s > 5),
]

for label, mask in thresholds:
    print(
        f"{label:14s}: "
        f"{int(mask.sum()):5,} "
        f"({100 * mask.mean():.2f}%)"
    )

print(f"\nMinimum LGD: {s.min():.4f}")
print(f"Maximum LGD: {s.max():.4f}")

# IMPORTANT:
# All choices and default must have compatible string dtype.

main["severity_group"] = np.select(
    [
        s < 0,
        (s >= 0) & (s <= 1),
        (s > 1) & (s <= 2),
        s > 2,
    ],
    [
        "negative",
        "0_to_100pct",
        "100_to_200pct",
        "above_200pct",
    ],
    default="unclassified",
)

if (main["severity_group"] == "unclassified").any():
    raise RuntimeError("Unclassified LGD observations found")


# ============================================================
# 6. Exposure and loss diagnostics
# ============================================================

print("\n=== 4. EAD and Loss by Severity Group ===")

groups = main.groupby("severity_group").agg(
    loans=("loan_id", "size"),
    median_ead=("upb_at_first90", "median"),
    median_actual_loss=("actual_loss", "median"),
    median_lgd=("lgd_signed", "median"),
)

print(groups.round(4).to_string())

groups.to_csv(
    OUT / "22_lgd_tail_summary.csv"
)

print("\n=== 5. Top 10 Highest LGD ===")

top = main.nlargest(10, "lgd_signed")[
    [
        "vintage",
        "exit_code",
        "upb_at_first90",
        "actual_loss",
        "lgd_signed",
        "months_to_exit",
    ]
]

print(top.round(4).to_string(index=False))

extreme = main.loc[
    (s < 0) | (s > 1)
].copy()

extreme.sort_values(
    "lgd_signed",
    ascending=False,
).to_csv(
    OUT / "22_lgd_extreme_cases.csv",
    index=False,
)


# ============================================================
# 7. Histogram + KDE
# ============================================================

view = s.loc[
    s.between(PLOT_MIN, PLOT_MAX)
]

outside_pct = (
    100 * (len(s) - len(view)) / len(s)
)

bins = 80
bin_width = (PLOT_MAX - PLOT_MIN) / bins

fig, ax = plt.subplots(figsize=(10, 6))

# Normalize using the full-sample denominator.
weights = np.full(
    len(view),
    1.0 / (len(s) * bin_width),
)

ax.hist(
    view,
    bins=bins,
    range=(PLOT_MIN, PLOT_MAX),
    weights=weights,
    alpha=0.55,
    label="Normalized histogram",
)

# Estimate the central density without allowing
# extreme outliers to dominate the KDE bandwidth.
if len(view) > 2 and view.std() > 0:
    kde = gaussian_kde(view.to_numpy(dtype=float))
    grid = np.linspace(PLOT_MIN, PLOT_MAX, 500)

    central_mass = len(view) / len(s)

    ax.plot(
        grid,
        kde(grid) * central_mass,
        linewidth=2,
        label="Central-range KDE (scaled)",
    )

median = s.median()

ax.axvline(
    median,
    linestyle="--",
    linewidth=1.2,
    label=f"Median = {median:.3f}",
)

ax.set_xlim(PLOT_MIN, PLOT_MAX)
ax.set_xlabel("Signed LGD")
ax.set_ylabel("Probability Density")
ax.set_title(
    "Historical LGD Density | Credit Resolution 02/03/09"
)

ax.legend()

ax.text(
    0.98,
    0.95,
    f"N = {len(s):,}\n"
    f"Mean = {s.mean():.3f}\n"
    f"Median = {median:.3f}\n"
    f"Outside view = {outside_pct:.2f}%",
    transform=ax.transAxes,
    ha="right",
    va="top",
)

fig.tight_layout()

fig.savefig(
    OUT / "22_lgd_histogram_kde.png",
    dpi=180,
)

plt.close(fig)


# ============================================================
# 8. Zoomed empirical CDF
# ============================================================

x = np.sort(s.to_numpy(dtype=float))
y = np.arange(1, len(x) + 1) / len(x)

p95 = s.quantile(0.95)

fig, ax = plt.subplots(figsize=(10, 6))

ax.step(
    x,
    y,
    where="post",
    linewidth=1.8,
    label="Empirical CDF",
)

ax.axvline(
    median,
    linestyle="--",
    label=f"Median = {median:.3f}",
)

ax.axvline(
    p95,
    linestyle="--",
    label=f"P95 = {p95:.3f}",
)

ax.axhline(0.50, linestyle=":", linewidth=1)
ax.axhline(0.95, linestyle=":", linewidth=1)

ax.set_xlim(PLOT_MIN, PLOT_MAX)
ax.set_ylim(0, 1.02)

ax.set_xlabel("Signed LGD")
ax.set_ylabel("Empirical Cumulative Probability")
ax.set_title("Historical LGD Empirical CDF (Zoomed)")
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT / "22_lgd_cdf_zoomed.png",
    dpi=180,
)

plt.close(fig)


# ============================================================
# 9. Final summary
# ============================================================

print("\n=== 6. Distribution Summary ===")
print(f"Mean:              {s.mean():.4f}")
print(f"Median:            {s.median():.4f}")
print(f"Standard deviation:{s.std():.4f}")
print(f"P95:               {p95:.4f}")
print(f"P99:               {s.quantile(0.99):.4f}")

print("\n=== 7. Saved Files ===")

for name in [
    "22_excluded_loans.csv",
    "22_lgd_extreme_cases.csv",
    "22_lgd_tail_summary.csv",
    "22_lgd_histogram_kde.png",
    "22_lgd_cdf_zoomed.png",
]:
    print(OUT / name)

print("\nTask 22 complete.")
