
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop" / "credit-portfolio-data"
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "13_default_to_loss_followup.parquet"

CODES = {
    "01": "PREPAID_OR_MATURED",
    "02": "CREDIT_RESOLUTION",
    "03": "CREDIT_RESOLUTION",
    "09": "CREDIT_RESOLUTION",
    "15": "WHOLE_LOAN_SALE",
    "16": "RPL_SECURITIZATION",
    "96": "DEFECT_TERMINATION",
}

OUTCOMES = [
    "CREDIT_RESOLUTION",
    "WHOLE_LOAN_SALE",
    "PREPAID_OR_MATURED",
    "RPL_SECURITIZATION",
    "DEFECT_TERMINATION",
    "OTHER_EXIT",
    "NO_EXIT_WITHIN_H",
]

LOSS_CODES = {"02", "03", "09", "15"}

COLS = [
    "loan_id", "vintage", "admitted_60",
    "exit_code", "exit_m", "last_m", "t0_m",
    "months_to_exit", "loss_status",
    "actual_loss", "has_defect",
    "exit_within_36", "exit_within_60",
    "known_36", "known_60",
    "status_36", "status_60",
    "ever_current_60", "three_current_60",
]

df = pd.read_parquet(INPUT, columns=COLS)

assert not df["loan_id"].duplicated().any(), \
    "Duplicate loan identifiers"

d = df.loc[df["admitted_60"].eq(True)].copy()

if d.empty:
    raise ValueError("No eligible 60M cohort")

if not d["known_60"].fillna(False).all():
    raise ValueError(
        "Unknown 60M outcomes found. "
        "Censoring requires separate handling."
    )

# --------------------------------------------------
# 1. Mutually exclusive outcomes
# --------------------------------------------------

for h in (36, 60):
    exited = d[f"exit_within_{h}"].fillna(False)

    category = (
        d["exit_code"]
        .map(CODES)
        .fillna("OTHER_EXIT")
    )

    d[f"outcome_{h}"] = category.where(
        exited, "NO_EXIT_WITHIN_H"
    )

# Once exited, a loan cannot re-enter this dataset.
early_exit = d["exit_within_36"].fillna(False)
late_exit = d["exit_within_60"].fillna(False)

if (early_exit & ~late_exit).any():
    raise ValueError(
        "A 36M exit is missing from 60M outcomes"
    )

if (
    early_exit
    & d["outcome_36"].ne(d["outcome_60"])
).any():
    raise ValueError(
        "Exit category changed between horizons"
    )

rows = []

for h in (36, 60):
    groups = [("ALL", d)] + [
        (str(v), g)
        for v, g in d.groupby("vintage")
    ]

    for vintage, g in groups:
        counts = g[f"outcome_{h}"].value_counts()

        if counts.sum() != len(g):
            raise ValueError(
                "Outcome accounting failed"
            )

        for outcome in OUTCOMES:
            n = int(counts.get(outcome, 0))

            rows.append({
                "vintage": vintage,
                "horizon_m": h,
                "outcome": outcome,
                "loans": n,
                "total": len(g),
                "share_pct": round(
                    100 * n / len(g), 3
                ),
            })

outcome_table = pd.DataFrame(rows)

outcome_table.to_csv(
    OUT / "14_resolution_outcomes.csv",
    index=False
)

# --------------------------------------------------
# 2. Loss disclosure by exit code and vintage
# --------------------------------------------------

coverage_rows = []

for h in (36, 60):
    selected = d.loc[
        d[f"exit_within_{h}"]
        & d["exit_code"].isin(LOSS_CODES)
    ]

    groups = [
        (str(v), code, g)
        for (v, code), g in selected.groupby(
            ["vintage", "exit_code"]
        )
    ]

    groups += [
        ("ALL", code, g)
        for code, g in selected.groupby("exit_code")
    ]

    for vintage, code, g in groups:
        n = len(g)
        status = g["loss_status"]

        missing = status.eq("missing")
        defect = g["has_defect"].fillna(False)

        coverage_rows.append({
            "vintage": vintage,
            "horizon_m": h,
            "exit_code": code,
            "applicable": n,
            "positive": int(
                status.eq("positive").sum()
            ),
            "zero": int(
                status.eq("zero").sum()
            ),
            "negative": int(
                status.eq("negative").sum()
            ),
            "missing": int(missing.sum()),
            "missing_with_defect": int(
                (missing & defect).sum()
            ),
            "missing_without_defect": int(
                (missing & ~defect).sum()
            ),
            "reported_pct": round(
                100 * (~missing).sum() / n, 2
            ),
            "median_exit_month": round(
                float(g["months_to_exit"].median()),
                2
            ),
            "p90_exit_month": round(
                float(
                    g["months_to_exit"].quantile(.90)
                ),
                2
            ),
        })

coverage = pd.DataFrame(coverage_rows)

coverage.to_csv(
    OUT / "14_loss_coverage_by_code.csv",
    index=False
)

# --------------------------------------------------
# 3. Transition from 36M to 60M
# --------------------------------------------------

transition = pd.crosstab(
    d["outcome_36"],
    d["outcome_60"]
)

transition.to_csv(
    OUT / "14_resolution_transition.csv"
)

# --------------------------------------------------
# 4. Integrity checks
# --------------------------------------------------

checks = []

def record(name, value):
    checks.append({
        "check": name,
        "count": int(value),
    })


record(
    "36M_exit_not_in_60M",
    (early_exit & ~late_exit).sum()
)

record(
    "60M_unknown_followup",
    (~d["known_60"].fillna(False)).sum()
)

record(
    "60M_no_exit_missing_status",
    (
        ~late_exit
        & d["status_60"].isna()
    ).sum()
)

record(
    "60M_no_exit_unavailable_status",
    (
        ~late_exit
        & d["status_60"].eq("unavailable")
    ).sum()
)

record(
    "36M_no_exit_missing_status",
    (
        ~early_exit
        & d["status_36"].isna()
    ).sum()
)

record(
    "exit_effective_after_last_report",
    (
        d["exit_m"].notna()
        & d["last_m"].notna()
        & (d["exit_m"] > d["last_m"])
    ).sum()
)

record(
    "positive_loss_nonapplicable_code",
    (
        ~d["exit_code"].isin(LOSS_CODES)
        & d["actual_loss"].gt(0)
    ).sum()
)

record(
    "ever_current_then_credit_exit_60M",
    (
        d["ever_current_60"].fillna(False)
        & late_exit
        & d["exit_code"].isin(
            {"02", "03", "09"}
        )
    ).sum()
)

record(
    "three_current_then_credit_exit_60M",
    (
        d["three_current_60"].fillna(False)
        & late_exit
        & d["exit_code"].isin(
            {"02", "03", "09"}
        )
    ).sum()
)

integrity = pd.DataFrame(checks)

integrity.to_csv(
    OUT / "14_integrity_checks.csv",
    index=False
)

# --------------------------------------------------
# 5. Compact terminal report
# --------------------------------------------------

print("\n=== COHORT ===")
print(f"Same 60M cohort: {len(d):,} loans")

print("\n=== EXCLUSIVE RESOLUTION OUTCOMES ===")

report = (
    outcome_table.loc[
        outcome_table["vintage"].eq("ALL")
    ]
    .pivot(
        index="outcome",
        columns="horizon_m",
        values="loans"
    )
    .reindex(OUTCOMES)
)

print(report.to_string())

print("\n=== LOSS COVERAGE BY CODE: 60M ===")

cols = [
    "exit_code",
    "applicable",
    "positive",
    "zero",
    "negative",
    "missing",
    "missing_with_defect",
    "reported_pct",
]

print(
    coverage.loc[
        coverage["vintage"].eq("ALL")
        & coverage["horizon_m"].eq(60),
        cols
    ].sort_values(
        "exit_code"
    ).to_string(index=False)
)

print("\n=== INTEGRITY CHECKS ===")
print(integrity.to_string(index=False))

print("\nSaved 4 CSV reports in outputs/")
