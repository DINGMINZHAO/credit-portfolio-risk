from pathlib import Path
import re
import zipfile

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

EVENTS = PROC / "03_loan_events_2006_2022.parquet"

VINTAGES = range(2006, 2012)

LOSS_CODES = {"02", "03", "09", "15"}

KNOWN_CODES = {
    "01", "02", "03", "09",
    "15", "16", "96"
}

# Release 47: 35 Performance columns.
# All indices are zero-based.
COL = {
    "loan_id": 0,
    "period": 1,
    "current_upb": 2,
    "dpd": 3,
    "defect_settlement": 6,
    "exit_code": 8,
    "exit_date": 9,
    "mi_recoveries": 13,
    "net_sale_proceeds": 14,
    "non_mi_recoveries": 15,
    "total_expenses": 16,
    "legal_costs": 17,
    "maintenance_costs": 18,
    "taxes_insurance": 19,
    "misc_expenses": 20,
    "actual_loss": 21,
    "zero_removal_upb": 26,
    "delinquent_interest": 27,
}

FORMULA = [
    "zero_removal_upb",
    "net_sale_proceeds",
    "delinquent_interest",
    "total_expenses",
    "mi_recoveries",
    "non_mi_recoveries",
]

EXPENSE_DETAIL = [
    "legal_costs",
    "maintenance_costs",
    "taxes_insurance",
    "misc_expenses",
]

TOLERANCE_USD = 0.10


# ============================================================
# 2. Helper functions
# ============================================================

def parse_ym(value):

    if value is None or pd.isna(value):
        return None

    s = str(value).strip()

    missing = {
        "", "nan", "nat", "none",
        "<na>", "null", "0", "000000"
    }

    if s.lower() in missing:
        return None

    if re.fullmatch(r"\d{6}(?:\.0+)?", s):

        result = int(s[:6])

    else:

        parsed = pd.to_datetime(
            s,
            errors="coerce"
        )

        if pd.isna(parsed):
            raise ValueError(
                f"Invalid month: {value!r}"
            )

        result = int(parsed.strftime("%Y%m"))

    year = result // 100
    month = result % 100

    if not (
        1900 <= year <= 2100
        and 1 <= month <= 12
    ):
        raise ValueError(
            f"Invalid month: {value!r}"
        )

    return result


def field(parts, name):

    return (
        parts[COL[name]]
        .strip()
        .decode("ascii")
    )


def amount(parts, name):

    s = field(parts, name).replace(",", "")

    if s.upper() in {
        "", "NULL", "NAN", "NA"
    }:
        return np.nan

    value = float(s)

    if not np.isfinite(value):
        raise ValueError(
            f"Non-finite value for {name}: {s}"
        )

    return value


def find_zip(year):

    matches = [
        p for p in RAW.rglob("*.zip")
        if p.name.lower() == f"sample_{year}.zip"
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one sample_{year}.zip "
            f"in {RAW}; found {len(matches)}"
        )

    return matches[0]


# ============================================================
# 3. Load loan identifiers
# ============================================================

def load_loan_index(con, year):

    path = str(EVENTS).replace("'", "''")

    source = con.execute(f"""
        SELECT
            loan_id,
            first_90_period

        FROM read_parquet('{path}')

        WHERE TRY_CAST(vintage AS INTEGER) = {year}
    """).fetchall()

    if len(source) != 50_000:

        raise ValueError(
            f"Vintage {year}: expected 50,000 loans; "
            f"found {len(source):,}"
        )

    states = {}

    for loan, expected90 in source:

        key = str(loan).strip().encode("ascii")

        if key in states:
            raise ValueError(
                f"Duplicate Loan ID in {year}"
            )

        states[key] = {

            "expected_first90":
                parse_ym(expected90),

            "raw_first90":
                None,

            "first90_upb":
                np.nan,

            "max_period":
                -1,

            "latest_raw":
                None,

            "ever_positive_loss":
                False,

            "duplicate_months":
                0,

            "n_records":
                0,
        }

    return states


# ============================================================
# 4. Scan each vintage
# ============================================================

def process_year(year, con):

    states = load_loan_index(con, year)

    path = find_zip(year)

    print(
        f"Processing {year} ...",
        flush=True
    )

    with zipfile.ZipFile(path) as archive:

        members = [
            name for name in archive.namelist()
            if Path(name).name.lower()
            == f"sample_perf_{year}.txt"
        ]

        if len(members) != 1:

            raise ValueError(
                f"{path.name}: expected one "
                "Performance text file"
            )

        with archive.open(members[0]) as file:

            for raw in file:

                parts = (
                    raw.rstrip(b"\r\n")
                    .split(b"|")
                )

                if len(parts) != 35:

                    raise ValueError(
                        f"{path.name}: expected 35 fields, "
                        f"found {len(parts)}"
                    )

                loan_key = parts[0].strip()

                state = states.get(loan_key)

                if state is None:

                    raise ValueError(
                        "Loan ID missing from event table"
                    )

                period = parse_ym(
                    parts[COL["period"]].decode("ascii")
                )

                if period is None:
                    raise ValueError(
                        "Missing reporting period"
                    )

                state["n_records"] += 1

                # --------------------------------------------
                # Capture earliest 90+ DPD and its UPB.
                # --------------------------------------------

                dpd = parts[COL["dpd"]].strip()

                if dpd.isdigit() and int(dpd) >= 3:

                    if (
                        state["raw_first90"] is None
                        or period < state["raw_first90"]
                    ):

                        state["raw_first90"] = period

                        state["first90_upb"] = amount(
                            parts,
                            "current_upb"
                        )

                # --------------------------------------------
                # Historical positive loss indicator.
                # Used only for reconciliation.
                # --------------------------------------------

                raw_loss = (
                    parts[COL["actual_loss"]].strip()
                )

                if (
                    raw_loss
                    and float(raw_loss) > 0
                ):

                    state["ever_positive_loss"] = True

                # --------------------------------------------
                # Keep the latest reporting record.
                # Never sum monthly Actual Loss values.
                # --------------------------------------------

                if period > state["max_period"]:

                    state["max_period"] = period

                    state["latest_raw"] = raw

                elif period == state["max_period"]:

                    state["duplicate_months"] += 1

                    state["latest_raw"] = raw

    # ========================================================
    # Convert latest records to loan-level rows
    # ========================================================

    rows = []
    missing = 0

    for loan_key, state in states.items():

        if state["latest_raw"] is None:

            missing += 1
            continue

        parts = (
            state["latest_raw"]
            .rstrip(b"\r\n")
            .split(b"|")
        )

        raw_code = field(
            parts,
            "exit_code"
        )

        code = (
            raw_code.zfill(2)
            if raw_code
            else ""
        )

        row = {

            "loan_id":
                loan_key.decode("ascii"),

            "vintage":
                year,

            "last_period":
                state["max_period"],

            "exit_code":
                code or None,

            "exit_date":
                parse_ym(field(parts, "exit_date")),

            "defect_settlement":
                parse_ym(
                    field(parts, "defect_settlement")
                ),

            "first90_event_table":
                state["expected_first90"],

            "first90_recomputed":
                state["raw_first90"],

            "upb_at_first90":
                state["first90_upb"],

            "ever_positive_loss_in_raw":
                state["ever_positive_loss"],

            "duplicate_latest_months":
                state["duplicate_months"],

            "monthly_records":
                state["n_records"],
        }

        monetary_fields = [
            "current_upb",
            *FORMULA,
            *EXPENSE_DETAIL,
            "actual_loss",
        ]

        for name in monetary_fields:

            row[name] = amount(
                parts,
                name
            )

        rows.append(row)

    if missing:

        raise ValueError(
            f"{year}: {missing} loans "
            "without Performance records"
        )

    result = pd.DataFrame(rows)

    positive = int(
        (result["actual_loss"] > 0).sum()
    )

    print(
        f"  Loans: {len(result):,} | "
        f"Final positive losses: {positive:,}",
        flush=True
    )

    return result


# ============================================================
# 5. Define audit indicators
# ============================================================

def derive_flags(df):

    d = df.copy()

    # Actual Loss applicability
    d["loss_applicable_exit"] = (
        d["exit_code"].isin(LOSS_CODES)
    )

    d["loss_reported"] = (
        d["actual_loss"].notna()
    )

    # Distinguish positive, zero, negative, missing.
    d["positive_loss"] = (
        d["actual_loss"] > 0
    )

    d["reported_zero"] = (
        d["actual_loss"] == 0
    )

    d["reported_gain"] = (
        d["actual_loss"] < 0
    )

    # --------------------------------------------
    # Reconcile earliest severe delinquency.
    # --------------------------------------------

    expected90 = (
        d["first90_event_table"]
        .fillna(-1)
        .astype(int)
    )

    recomputed90 = (
        d["first90_recomputed"]
        .fillna(-1)
        .astype(int)
    )

    d["first90_matches"] = (
        expected90 == recomputed90
    )

    # --------------------------------------------
    # Reconstruct Actual Loss only when all
    # six components have reported values.
    # --------------------------------------------

    d["formula_checkable"] = (
        d["loss_reported"]
        & d[FORMULA].notna().all(axis=1)
    )

    d["formula_rebuilt"] = (
        d[FORMULA].sum(
            axis=1,
            min_count=len(FORMULA)
        )
    )

    d["formula_gap_usd"] = (
        d["actual_loss"]
        - d["formula_rebuilt"]
    )

    d["formula_mismatch"] = (
        d["formula_checkable"]
        & (
            d["formula_gap_usd"].abs()
            > TOLERANCE_USD
        )
    )

    # --------------------------------------------
    # Check Total Expenses against components.
    # --------------------------------------------

    d["expense_checkable"] = (
        d["total_expenses"].notna()
        & d[EXPENSE_DETAIL].notna().all(axis=1)
    )

    d["expense_rebuilt"] = (
        d[EXPENSE_DETAIL].sum(
            axis=1,
            min_count=len(EXPENSE_DETAIL)
        )
    )

    d["expense_mismatch"] = (
        d["expense_checkable"]
        & (
            (
                d["total_expenses"]
                - d["expense_rebuilt"]
            ).abs()
            > TOLERANCE_USD
        )
    )

    return d


# ============================================================
# 6. Aggregate results
# ============================================================

def summarize(df):

    summary_rows = []
    audit_rows = []

    for year, g in df.groupby(
        "vintage",
        sort=True
    ):

        summary_rows.append({

            "vintage":
                year,

            "loans":
                len(g),

            "exited":
                int(g["exit_code"].notna().sum()),

            "loss_applicable_exits":
                int(g["loss_applicable_exit"].sum()),

            "actual_loss_reported":
                int(g["loss_reported"].sum()),

            "positive_losses":
                int(g["positive_loss"].sum()),

            "reported_zero":
                int(g["reported_zero"].sum()),

            "reported_gains":
                int(g["reported_gain"].sum()),

            "positive_loss_usd":
                g.loc[
                    g["positive_loss"],
                    "actual_loss"
                ].sum(),

            "positive_loss_with_first90":
                int((
                    g["positive_loss"]
                    & g["first90_recomputed"].notna()
                ).sum()),

            "positive_loss_with_first90_upb":
                int((
                    g["positive_loss"]
                    & g["upb_at_first90"].gt(0)
                ).sum()),
        })

        audit_rows.append({

            "vintage":
                year,

            "first90_timing_mismatches":
                int((~g["first90_matches"]).sum()),

            "duplicate_latest_months":
                int(g["duplicate_latest_months"].sum()),

            "loss_outside_applicable_codes":
                int((
                    g["loss_reported"]
                    & ~g["loss_applicable_exit"]
                ).sum()),

            "applicable_exit_no_loss":
                int((
                    g["loss_applicable_exit"]
                    & ~g["loss_reported"]
                ).sum()),

            "formula_checkable":
                int(g["formula_checkable"].sum()),

            "formula_mismatch":
                int(g["formula_mismatch"].sum()),

            "expense_checkable":
                int(g["expense_checkable"].sum()),

            "expense_mismatch":
                int(g["expense_mismatch"].sum()),

            "ever_positive_but_final_not_positive":
                int((
                    g["ever_positive_loss_in_raw"]
                    & ~g["positive_loss"]
                ).sum()),

            "unexpected_exit_codes":
                int((
                    g["exit_code"].notna()
                    & ~g["exit_code"].isin(KNOWN_CODES)
                ).sum()),

            "positive_loss_no_first90":
                int((
                    g["positive_loss"]
                    & g["first90_recomputed"].isna()
                ).sum()),

            "positive_loss_missing_removal_upb":
                int((
                    g["positive_loss"]
                    & ~g["zero_removal_upb"].gt(0)
                ).sum()),
        })

    summary = pd.DataFrame(summary_rows)

    denominator = (
        summary["loss_applicable_exits"]
        .replace(0, np.nan)
    )

    summary["reported_pct_of_applicable"] = (
        100
        * summary["actual_loss_reported"]
        / denominator
    )

    audit = pd.DataFrame(audit_rows)

    # ========================================================
    # Breakdown by Zero Balance Code
    # ========================================================

    exit_rows = []

    for (year, code), g in df.groupby(
        ["vintage", "exit_code"],
        dropna=False
    ):

        exit_rows.append({

            "vintage":
                year,

            "exit_code":
                code if isinstance(code, str)
                else "NO_EXIT",

            "loans":
                len(g),

            "loss_reported":
                int(g["loss_reported"].sum()),

            "positive_losses":
                int(g["positive_loss"].sum()),

            "zero_losses":
                int(g["reported_zero"].sum()),

            "negative_gains":
                int(g["reported_gain"].sum()),

            "positive_loss_usd":
                g.loc[
                    g["positive_loss"],
                    "actual_loss"
                ].sum(),
        })

    exit_table = (
        pd.DataFrame(exit_rows)
        .sort_values(
            ["vintage", "exit_code"]
        )
    )

    return summary, audit, exit_table


# ============================================================
# 7. Visualizations
# ============================================================

def draw_figures(df, summary):

    # Figure 1: positive loss count by vintage
    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    ax.bar(
        summary["vintage"].astype(str),
        summary["positive_losses"]
    )

    ax.set(
        title=(
            "Positive Reported Actual Loss "
            "by Origination Vintage"
        ),
        xlabel="Origination vintage",
        ylabel="Number of loans",
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "11_positive_loss_by_vintage.png",
        dpi=160
    )

    plt.close(fig)

    # Figure 2: positive Actual Loss distribution
    positive = (
        df.loc[
            df["positive_loss"],
            "actual_loss"
        ].to_numpy()
    )

    if len(positive):

        fig, ax = plt.subplots(
            figsize=(9, 4.8)
        )

        ax.hist(
            positive,
            bins=50
        )

        ax.set(
            title=(
                "Distribution of Positive "
                "Reported Actual Loss"
            ),
            xlabel="Positive Actual Loss (USD)",
            ylabel="Number of loans",
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        fig.tight_layout()

        fig.savefig(
            FIG / "11_positive_loss_distribution.png",
            dpi=160
        )

        plt.close(fig)


# ============================================================
# 8. Main
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    FIG.mkdir(
        parents=True,
        exist_ok=True
    )

    if not EVENTS.is_file():
        raise FileNotFoundError(EVENTS)

    con = duckdb.connect()

    con.execute("SET threads=2")
    con.execute("SET memory_limit='1GB'")

    try:

        frames = [
            process_year(year, con)
            for year in VINTAGES
        ]

    finally:

        con.close()

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df = derive_flags(df)

    if (
        len(df) != 300_000
        or df["loan_id"].duplicated().any()
    ):

        raise AssertionError(
            "Expected 300,000 distinct loans"
        )

    summary, audit, by_exit = summarize(df)

    # ========================================================
    # Save results
    # ========================================================

    df.to_parquet(
        PROC / "11_historical_loss_2006_2011.parquet",
        index=False
    )

    summary.to_csv(
        OUT / "11_loss_summary_by_vintage.csv",
        index=False,
        float_format="%.4f"
    )

    audit.to_csv(
        OUT / "11_loss_data_audit.csv",
        index=False
    )

    by_exit.to_csv(
        OUT / "11_loss_by_exit_code.csv",
        index=False,
        float_format="%.4f"
    )

    draw_figures(
        df,
        summary
    )

    # ========================================================
    # Short terminal output
    # ========================================================

    print(
        "\n=== Historical Actual Loss by Vintage ==="
    )

    cols = [
        "vintage",
        "loans",
        "loss_applicable_exits",
        "actual_loss_reported",
        "positive_losses",
        "positive_loss_with_first90_upb",
    ]

    print(
        summary[cols].to_string(index=False)
    )

    print(
        "\n=== Data Audit ==="
    )

    audit_cols = [
        "vintage",
        "first90_timing_mismatches",
        "applicable_exit_no_loss",
        "formula_checkable",
        "formula_mismatch",
        "ever_positive_but_final_not_positive",
    ]

    print(
        audit[audit_cols].to_string(index=False)
    )

    print(
        "\nSaved 1 Parquet, 3 CSV files, "
        "and 2 figures."
    )


if __name__ == "__main__":
    main()