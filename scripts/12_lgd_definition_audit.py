from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data/processed"
    / "11_historical_loss_2006_2011.parquet"
)

OUT = ROOT / "outputs"
FIG = OUT / "figures"
PROC = ROOT / "data/processed"

APPLICABLE = {"02", "03", "09", "15"}

CUTOFF = 202603
EXPECTED = 300_000

FIELDS = [
    "loan_id",
    "vintage",
    "exit_code",
    "exit_date",
    "defect_settlement",
    "first90_recomputed",
    "upb_at_first90",
    "zero_removal_upb",
    "actual_loss",
    "last_period",
]


# ============================================================
# 2. Helpers
# ============================================================

def as_month_index(s, name):

    numeric = pd.to_numeric(
        s, errors="coerce"
    )

    yy = numeric // 100
    mm = numeric % 100

    bad = numeric.notna() & (
        (numeric != np.floor(numeric))
        | ~yy.between(1900, 2100)
        | ~mm.between(1, 12)
    )

    if bad.any():
        raise ValueError(
            f"Invalid YYYYMM in {name}: "
            f"{int(bad.sum())} rows"
        )

    return yy * 12 + (mm - 1)


def group_summary(d, group_columns):

    return (
        d.groupby(
            group_columns,
            dropna=False,
            observed=True,
        )
        .agg(
            loans=("loan_id", "size"),

            reported=("loss_reported", "sum"),
            missing=("missing_loss", "sum"),

            positive=("positive_loss", "sum"),
            zero=("zero_loss", "sum"),
            negative=("negative_loss", "sum"),

            has_defect=("has_defect", "sum"),
            near_cutoff=("near_cutoff", "sum"),

            usable_first90_balance=(
                "valid_first90", "sum"
            ),

            usable_exit_balance=(
                "valid_exit_balance", "sum"
            ),

            usable_both_balances=(
                "valid_both", "sum"
            ),
        )
        .reset_index()
    )


def distribution_row(g, label):

    a = g.loc[
        g["valid_both"],
        "exit_over_first90",
    ].dropna()

    return {
        "exit_code": label,

        "n_valid_pairs": len(a),

        "median_exit_over_first90":
            a.median(),

        "p10": a.quantile(0.10),
        "p25": a.quantile(0.25),
        "p75": a.quantile(0.75),
        "p90": a.quantile(0.90),

        "pct_exit_balance_above_first90":
            100 * (a > 1).mean()
            if len(a) else np.nan,

        "pct_exit_balance_above_first90_by_5pct":
            100 * (a > 1.05).mean()
            if len(a) else np.nan,
    }


# ============================================================
# 3. Main
# ============================================================

def main():

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Read only the required columns
    # --------------------------------------------------------

    d = pd.read_parquet(
        SOURCE,
        columns=FIELDS,
    )

    if (
        len(d) != EXPECTED
        or d["loan_id"].isna().any()
        or d["loan_id"].duplicated().any()
    ):
        raise ValueError(
            "Expected 300,000 distinct loan IDs"
        )

    if set(
        d["vintage"].dropna().astype(int)
    ) != set(range(2006, 2012)):
        raise ValueError(
            "Unexpected vintage range"
        )

    # --------------------------------------------------------
    # Normalize exit codes and monetary amounts
    # --------------------------------------------------------

    d["exit_code"] = (
        d["exit_code"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .str.replace(
            r"\.0+$", "", regex=True
        )
        .str.zfill(2)
    )

    for field in [
        "upb_at_first90",
        "zero_removal_upb",
        "actual_loss",
    ]:
        d[field] = pd.to_numeric(
            d[field],
            errors="raise",
        )

    # --------------------------------------------------------
    # Audit settlement and reporting dates
    # --------------------------------------------------------

    d["has_defect"] = (
        d["defect_settlement"].notna()
    )

    exit_m = as_month_index(
        d["exit_date"],
        "exit_date",
    )

    first90_m = as_month_index(
        d["first90_recomputed"],
        "first90_recomputed",
    )

    as_month_index(
        d["last_period"],
        "last_period",
    )

    cutoff_m = (
        (CUTOFF // 100) * 12
        + (CUTOFF % 100) - 1
    )

    months_from_cutoff = (
        cutoff_m - exit_m
    )

    # Screening flag, not a definitive disclosure rule.
    d["near_cutoff"] = (
        months_from_cutoff
        .between(0, 3)
        .fillna(False)
    )

    # --------------------------------------------------------
    # Actual Loss applicability and status
    # --------------------------------------------------------

    d["loss_applicable"] = (
        d["exit_code"]
        .isin(APPLICABLE)
        .fillna(False)
    )

    d["loss_reported"] = (
        d["actual_loss"].notna()
    )

    unexpected = (
        d["loss_reported"]
        & ~d["loss_applicable"]
    )

    if unexpected.any():
        raise ValueError(
            "Actual Loss outside applicable "
            "exit codes: "
            f"{int(unexpected.sum())}"
        )

    d["missing_loss"] = (
        d["loss_applicable"]
        & ~d["loss_reported"]
    )

    d["positive_loss"] = (
        d["loss_applicable"]
        & d["actual_loss"].gt(0)
    )

    d["zero_loss"] = (
        d["loss_applicable"]
        & d["actual_loss"].eq(0)
    )

    d["negative_loss"] = (
        d["loss_applicable"]
        & d["actual_loss"].lt(0)
    )

    d["loss_status"] = np.select(
        [
            d["missing_loss"],
            d["positive_loss"],
            d["zero_loss"],
            d["negative_loss"],
        ],
        [
            "missing",
            "positive",
            "zero",
            "negative",
        ],
        default="not_applicable",
    )

    # --------------------------------------------------------
    # Explain missing Actual Loss
    # --------------------------------------------------------

    # Reasons are assigned hierarchically:
    # 1. Known defect settlement
    # 2. Near performance cutoff
    # 3. Requires further investigation

    d["missing_reason"] = np.select(
        [
            d["missing_loss"]
            & d["has_defect"],

            d["missing_loss"]
            & d["near_cutoff"],

            d["missing_loss"],
        ],
        [
            "defect_settlement",
            "near_cutoff_review",
            "unexplained_review",
        ],
        default="not_missing",
    )

    # --------------------------------------------------------
    # Validate exposure denominators
    # --------------------------------------------------------

    d["valid_first90"] = (
        d["upb_at_first90"]
        .gt(0).fillna(False)
        & first90_m.notna()
    )

    d["valid_exit_balance"] = (
        d["zero_removal_upb"]
        .gt(0).fillna(False)
    )

    d["valid_both"] = (
        d["loss_reported"]
        & d["loss_applicable"]
        & d["valid_first90"]
        & d["valid_exit_balance"]
    )

    d["first90_after_exit"] = (
        first90_m.notna()
        & exit_m.notna()
        & (first90_m > exit_m)
    )

    # --------------------------------------------------------
    # Compare first-90 UPB and exit UPB
    # --------------------------------------------------------

    first90_base = (
        d["upb_at_first90"]
        .where(d["valid_both"])
    )

    exit_base = (
        d["zero_removal_upb"]
        .where(d["valid_both"])
    )

    d["exit_over_first90"] = (
        d["zero_removal_upb"]
        / first90_base
    )

    # Diagnostic ratios only, NOT final LGD.
    d["signed_loss_over_first90"] = (
        d["actual_loss"]
        / first90_base
    )

    d["signed_loss_over_exit_balance"] = (
        d["actual_loss"]
        / exit_base
    )

    # ========================================================
    # 4. Aggregated audit tables
    # ========================================================

    a = d.loc[
        d["loss_applicable"]
    ].copy()

    # Table 1: Coverage by vintage and exit code

    coverage = group_summary(
        a,
        ["vintage", "exit_code"],
    )

    coverage["pct_reported"] = (
        100
        * coverage["reported"]
        / coverage["loans"]
    )

    coverage.to_csv(
        OUT / "12_coverage_by_vintage_exit.csv",
        index=False,
        float_format="%.5f",
    )

    # Table 2: Missing loss reasons

    missing = (
        a.loc[a["missing_loss"]]
        .groupby(
            [
                "vintage",
                "exit_code",
                "missing_reason",
            ],
            observed=True,
        )
        .size()
        .rename("loans")
        .reset_index()
    )

    missing.to_csv(
        OUT / "12_missing_loss_reasons.csv",
        index=False,
    )

    # Table 3: Positive / zero / negative losses

    statuses = (
        a.groupby(
            [
                "vintage",
                "exit_code",
                "loss_status",
            ],
            observed=True,
        )
        .size()
        .rename("loans")
        .reset_index()
    )

    statuses.to_csv(
        OUT / "12_loss_status_counts.csv",
        index=False,
    )

    # Table 4: Balance comparison

    ratio_rows = [
        distribution_row(g, str(code))
        for code, g in a.groupby(
            "exit_code",
            observed=True,
        )
    ]

    ratio_rows.append(
        distribution_row(a, "ALL")
    )

    ratio_table = pd.DataFrame(
        ratio_rows
    )

    ratio_table.to_csv(
        OUT / "12_balance_ratio_diagnostics.csv",
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Save detailed loan-level audit candidates
    # --------------------------------------------------------

    audit_cols = [
        "loan_id",
        "vintage",
        "exit_code",
        "exit_date",
        "defect_settlement",
        "actual_loss",
        "loss_status",
        "missing_reason",
        "has_defect",
        "near_cutoff",
        "first90_recomputed",
        "upb_at_first90",
        "zero_removal_upb",
        "valid_first90",
        "valid_exit_balance",
        "valid_both",
        "first90_after_exit",
        "exit_over_first90",
        "signed_loss_over_first90",
        "signed_loss_over_exit_balance",
    ]

    a[audit_cols].to_parquet(
        PROC / "12_lgd_definition_audit.parquet",
        index=False,
    )

    # ========================================================
    # 5. Visualization
    # ========================================================

    graph = ratio_table.loc[
        (ratio_table["exit_code"] != "ALL")
        & (ratio_table["n_valid_pairs"] > 0)
    ]

    if not graph.empty:

        fig, ax = plt.subplots(
            figsize=(8, 4.5)
        )

        y = np.arange(len(graph))

        med = graph[
            "median_exit_over_first90"
        ].to_numpy(dtype=float)

        lo = graph["p10"].to_numpy(
            dtype=float
        )

        hi = graph["p90"].to_numpy(
            dtype=float
        )

        ax.errorbar(
            med,
            y,
            xerr=[
                med - lo,
                hi - med,
            ],
            fmt="o",
            capsize=4,
        )

        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

        ax.set_yticks(
            y,
            graph["exit_code"].tolist(),
        )

        ax.set_xlabel(
            "Exit UPB / UPB at first 90+ "
            "(median, 10th-90th pct)"
        )

        ax.set_ylabel(
            "Zero Balance Code"
        )

        ax.set_title(
            "Balance timing diagnostic: "
            "reported-loss loans"
        )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        fig.tight_layout()

        fig.savefig(
            FIG / "12_balance_ratio_by_exit_code.png",
            dpi=160,
        )

        plt.close(fig)

    # ========================================================
    # 6. Short terminal output
    # ========================================================

    pooled = group_summary(
        a,
        ["exit_code"],
    )

    pooled["pct_reported"] = (
        100
        * pooled["reported"]
        / pooled["loans"]
    )

    print(
        "\n=== Actual Loss Coverage by Exit Code ==="
    )

    print(
        pooled[
            [
                "exit_code",
                "loans",
                "reported",
                "missing",
                "positive",
                "zero",
                "negative",
                "pct_reported",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n=== Missing Actual Loss: "
        "Explanation Candidates ==="
    )

    print(
        a.loc[
            a["missing_loss"],
            "missing_reason",
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== Balance Audit: "
        "All Applicable Exits ==="
    )

    print(
        "Reported Actual Loss: "
        f"{int(a['loss_reported'].sum()):,}"
    )

    print(
        "Reported loss with first-90 UPB > 0: "
        f"{int((a['loss_reported'] & a['valid_first90']).sum()):,}"
    )

    print(
        "Reported loss with exit UPB > 0: "
        f"{int((a['loss_reported'] & a['valid_exit_balance']).sum()):,}"
    )

    print(
        "Reported loss with both UPBs > 0: "
        f"{int(a['valid_both'].sum()):,}"
    )

    print(
        "Positive losses without valid first-90 UPB: "
        f"{int((a['positive_loss'] & ~a['valid_first90']).sum()):,}"
    )

    print(
        "First 90+ later than exit date: "
        f"{int(a['first90_after_exit'].sum()):,}"
    )

    print(
        "\n=== Exit UPB / First-90 UPB ==="
    )

    print(
        ratio_table
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nSaved 4 CSV files, 1 Parquet, "
        "and 1 figure "
        "(if valid balance pairs exist)."
    )


if __name__ == "__main__":
    main()