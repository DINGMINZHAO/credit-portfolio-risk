
"""
EDA Phase 2 - Task 01
Freddie Mac SFLLD Release 47

Extract first observed 90+ DPD and loan exit events.
Process one origination vintage at a time.
"""

from pathlib import Path
from zipfile import ZipFile
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 150_000

# Release 47: zero-based column indices
ORIG_COLS = {
    1: "first_payment",
    19: "loan_id",
}

PERF_COLS = {
    0: "loan_id",
    1: "period",
    3: "delinquency",
    8: "zero_code",
    9: "zero_effective",
}

EXIT_GROUPS = {
    "01": "Prepaid / matured",
    "02": "Credit disposition",
    "03": "Credit disposition",
    "09": "Credit disposition",
    "15": "Transfer / securitization",
    "16": "Transfer / securitization",
    "96": "Defect-related exit",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def month_serial(series):
    """Convert YYYYMM to a calendar-month index."""

    values = series.astype("string").str.strip()

    valid = values.str.fullmatch(
        r"\d{4}(0[1-9]|1[0-2])"
    ).fillna(False)

    numeric = pd.to_numeric(
        values.where(valid),
        errors="coerce"
    )

    return (
        numeric // 100 * 12 + numeric % 100
    ).astype("Float64")


def check_layout(zip_file, member, expected_columns):
    """Verify file layout before processing."""

    with zip_file.open(member) as f:
        first_line = (
            f.readline()
            .decode("utf-8-sig")
            .rstrip("\r\n")
        )

    actual_columns = first_line.count("|") + 1

    if actual_columns != expected_columns:
        raise ValueError(
            f"{member}: expected {expected_columns} "
            f"columns, found {actual_columns}"
        )


# --------------------------------------------------
# Main extraction
# --------------------------------------------------

def process_vintage(year):

    archive = ROOT / "data" / "raw" / f"sample_{year}.zip"

    processed_dir = ROOT / "data" / "processed"
    output_dir = ROOT / "outputs"
    figure_dir = output_dir / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    orig_name = f"sample_orig_{year}.txt"
    perf_name = f"sample_perf_{year}.txt"

    print(f"\nProcessing vintage {year}...", flush=True)

    with ZipFile(archive) as zf:

        check_layout(zf, orig_name, 31)
        check_layout(zf, perf_name, 35)

        # ------------------------------------------
        # Load loan origination information
        # ------------------------------------------

        with zf.open(orig_name) as f:

            loans = pd.read_csv(
                f,
                sep="|",
                header=None,
                usecols=list(ORIG_COLS),
                dtype=str,
                keep_default_na=False,
            ).rename(columns=ORIG_COLS)

        for col in loans.columns:
            loans[col] = loans[col].str.strip()

        if (
            loans["loan_id"].eq("").any()
            or loans["loan_id"].duplicated().any()
        ):
            raise ValueError(
                "Blank or duplicated loan identifiers."
            )

        loan_index = pd.Index(loans["loan_id"])
        n_loans = len(loans)

        # One array position per loan
        first_period = np.full(
            n_loans, 999999, dtype=np.int32
        )
        last_period = np.zeros(
            n_loans, dtype=np.int32
        )
        first_90 = np.full(
            n_loans, 999999, dtype=np.int32
        )
        record_counts = np.zeros(
            n_loans, dtype=np.int32
        )

        exit_reporting = np.zeros(
            n_loans, dtype=np.int32
        )
        exit_codes = [""] * n_loans
        exit_dates = [""] * n_loans

        ra_rows = 0
        xx_rows = 0
        conflicting_exits = 0

        # ------------------------------------------
        # Stream monthly performance records
        # ------------------------------------------

        with zf.open(perf_name) as f:

            reader = pd.read_csv(
                f,
                sep="|",
                header=None,
                usecols=list(PERF_COLS),
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            )

            for batch_number, chunk in enumerate(
                reader, start=1
            ):

                chunk = chunk.rename(
                    columns=PERF_COLS
                )

                for col in chunk.columns:
                    chunk[col] = chunk[col].str.strip()

                positions = loan_index.get_indexer(
                    chunk["loan_id"]
                )

                if (positions < 0).any():
                    raise ValueError(
                        f"Batch {batch_number}: "
                        "unmatched loan identifiers."
                    )

                valid_periods = (
                    chunk["period"]
                    .str.fullmatch(
                        r"\d{4}(0[1-9]|1[0-2])"
                    )
                )

                if not valid_periods.all():
                    raise ValueError(
                        f"Batch {batch_number}: "
                        "invalid reporting period."
                    )

                periods = chunk["period"].to_numpy(
                    dtype=np.int32
                )

                # First and last observed month
                np.minimum.at(
                    first_period, positions, periods
                )
                np.maximum.at(
                    last_period, positions, periods
                )

                # Number of monthly observations
                np.add.at(
                    record_counts, positions, 1
                )

                # First observed 90+ DPD
                delinquency_num = pd.to_numeric(
                    chunk["delinquency"],
                    errors="coerce"
                )

                severe = (
                    delinquency_num.between(3, 99)
                    .to_numpy()
                )

                np.minimum.at(
                    first_90,
                    positions[severe],
                    periods[severe]
                )

                ra_rows += int(
                    chunk["delinquency"].eq("RA").sum()
                )
                xx_rows += int(
                    chunk["delinquency"].eq("XX").sum()
                )

                # Zero Balance Codes
                raw_codes = chunk["zero_code"]

                codes = raw_codes.where(
                    raw_codes.eq(""),
                    raw_codes.str.zfill(2)
                )

                has_exit = codes.ne("").to_numpy()

                exit_positions = positions[has_exit]
                exit_periods = periods[has_exit]
                exit_values = codes.to_numpy()[has_exit]

                effective_dates = (
                    chunk["zero_effective"]
                    .to_numpy()[has_exit]
                )

                # Usually a single exit event per loan.
                # Keep the latest reported exit record.
                for pos, period, code, effective in zip(
                    exit_positions,
                    exit_periods,
                    exit_values,
                    effective_dates,
                ):

                    if (
                        exit_reporting[pos] == period
                        and exit_codes[pos] != code
                    ):
                        conflicting_exits += 1

                    if period >= exit_reporting[pos]:
                        exit_reporting[pos] = period
                        exit_codes[pos] = code
                        exit_dates[pos] = effective

                if batch_number % 10 == 0:
                    print(
                        f"  Processed "
                        f"{int(record_counts.sum()):,} rows",
                        flush=True
                    )

    # ----------------------------------------------
    # Build loan-level event table
    # ----------------------------------------------

    loans["vintage"] = year
    loans["monthly_records"] = record_counts

    loans["first_period"] = (
        pd.Series(first_period)
        .mask(first_period == 999999)
        .astype("Int32")
    )

    loans["last_period"] = (
        pd.Series(last_period)
        .mask(last_period == 0)
        .astype("Int32")
    )

    loans["first_90_period"] = (
        pd.Series(first_90)
        .mask(first_90 == 999999)
        .astype("Int32")
    )

    loans["zero_code"] = exit_codes
    loans["zero_effective"] = exit_dates

    loans["exit_reporting_period"] = (
        pd.Series(exit_reporting)
        .mask(exit_reporting == 0)
        .astype("Int32")
    )

    loans["exit_group"] = (
        loans["zero_code"]
        .map(EXIT_GROUPS)
        .fillna("Other / unknown exit")
    )

    loans.loc[
        loans["zero_code"].eq(""),
        "exit_group"
    ] = "No reported exit"

    # Calculate timing from first scheduled payment
    first_payment_month = month_serial(
        loans["first_payment"]
    )

    default_month = month_serial(
        loans["first_90_period"]
    )

    months_to_default = (
        default_month - first_payment_month + 1
    )

    loans["months_to_first_90"] = (
        months_to_default.astype("Int32")
    )

    loans["ever_90"] = (
        loans["first_90_period"].notna()
    )

    # ----------------------------------------------
    # Data quality checks
    # ----------------------------------------------

    no_performance = int(
        (record_counts == 0).sum()
    )

    missing_first_payment = int(
        first_payment_month.isna().sum()
    )

    invalid_default_timing = int(
        months_to_default.lt(1).fillna(False).sum()
    )

    missing_exit_date = int(
        (
            loans["zero_code"].ne("")
            & month_serial(
                loans["zero_effective"]
            ).isna()
        ).sum()
    )

    unknown_exit_codes = int(
        (
            loans["zero_code"].ne("")
            & ~loans["zero_code"].isin(EXIT_GROUPS)
        ).sum()
    )

    # ----------------------------------------------
    # Save Parquet
    # ----------------------------------------------

    events_path = (
        processed_dir
        / f"02_loan_events_{year}.parquet"
    )

    loans.to_parquet(
        events_path,
        index=False
    )

    # Exit code summary
    exit_counts = (
        loans["zero_code"]
        .replace("", "NO_CODE")
        .value_counts()
        .rename_axis("zero_code")
        .reset_index(name="loans")
    )

    exit_counts.to_csv(
        output_dir / f"02_exit_codes_{year}.csv",
        index=False
    )

    # ----------------------------------------------
    # Figure 1: Default Timing
    # ----------------------------------------------

    valid_months = (
        loans["months_to_first_90"].dropna()
    )

    valid_months = valid_months[
        valid_months.ge(1)
    ]

    if not valid_months.empty:

        upper = int(
            np.ceil(valid_months.max() / 12) * 12
        )

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        ax.hist(
            valid_months,
            bins=np.arange(0, upper + 13, 12),
            edgecolor="white"
        )

        ax.set(
            xlabel="Month index since first payment",
            ylabel="Number of loans",
            title=f"{year}: First Observed 90+ DPD"
        )

        fig.tight_layout()

        fig.savefig(
            figure_dir / f"02_first90_timing_{year}.png",
            dpi=160
        )

        plt.close(fig)

    # ----------------------------------------------
    # Figure 2: Loan Exit Codes
    # ----------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.barh(
        exit_counts["zero_code"],
        exit_counts["loans"]
    )

    ax.set(
        xlabel="Number of loans",
        ylabel="Zero Balance Code",
        title=f"{year}: Loan Exit Distribution"
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir / f"02_exit_codes_{year}.png",
        dpi=160
    )

    plt.close(fig)

    # ----------------------------------------------
    # Terminal summary
    # ----------------------------------------------

    print(f"\n=== Vintage {year} ===")

    print(f"Loans: {n_loans:,}")

    print(
        f"Performance rows: "
        f"{int(record_counts.sum()):,}"
    )

    print(
        f"Ever 90+ DPD: "
        f"{int(loans['ever_90'].sum()):,}"
    )

    print(
        f"Observed Ever 90+ rate: "
        f"{loans['ever_90'].mean():.2%}"
    )

    print(
        f"Loans with exit: "
        f"{int(loans['zero_code'].ne('').sum()):,}"
    )

    print(
        f"Max reporting period: "
        f"{int(last_period.max())}"
    )

    print("\nExit Codes:")
    print(exit_counts.to_string(index=False))

    print("\nData Quality:")
    print(f"No performance: {no_performance:,}")
    print(f"Missing first payment: {missing_first_payment:,}")
    print(f"Invalid default timing: {invalid_default_timing:,}")
    print(f"Missing exit date: {missing_exit_date:,}")
    print(f"Unknown exit codes: {unknown_exit_codes:,}")
    print(f"Conflicting exit rows: {conflicting_exits:,}")
    print(f"RA rows: {ra_rows:,}")
    print(f"XX rows: {xx_rows:,}")

    print("\nProcessing completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=2006
    )

    args = parser.parse_args()

    process_vintage(args.year)
