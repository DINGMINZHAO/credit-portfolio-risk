from pathlib import Path
from collections import Counter
import zipfile
import re

import numpy as np
import pandas as pd
import duckdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

EVENTS = PROC / "03_loan_events_2006_2022.parquet"
HORIZON = PROC / "07_horizon_events.parquet"
REFERENCE = OUT / "08_vintage_monthly_first90.csv"

VINTAGES = (2017, 2018, 2019)

START_EVENT = 201901
END_EVENT = 202112

EXPECTED_COLS = 35

# Zero-based positions in Release 47 Performance files
IDX_LOAN = 0
IDX_PERIOD = 1
IDX_DPD = 3
IDX_DISASTER = 28
IDX_ASSISTANCE = 29

METRICS = [
    "f_event",
    "f_prior3",
    "f_window",
    "disaster_event",
    "disaster_window",
    "either_window",
    "other_assistance_event",
    "pre3_complete",
    "f_window_complete",
    "either_window_complete",
]


# ============================================================
# Helpers
# ============================================================

def ym(value):
    """
    Convert a YYYYMM or date value to integer YYYYMM.

    Recognized missing values return None.
    Unexpected invalid dates raise ValueError.
    """

    # Handle None, NaN, NaT, pd.NA
    if value is None or pd.isna(value):
        return None

    s = str(value).strip()

    missing = {
        "",
        "nan",
        "nat",
        "none",
        "null",
        "<na>",
        "0",
        "000000",
    }

    if s.lower() in missing:
        return None

    # Handle YYYYMM or YYYYMM.0
    if re.fullmatch(r"\d{6}(?:\.0+)?", s):

        result = int(s[:6])

    else:

        parsed = pd.to_datetime(
            s,
            errors="coerce",
        )

        if pd.isna(parsed):
            raise ValueError(
                f"Invalid non-missing date: {value!r}"
            )

        result = int(parsed.strftime("%Y%m"))

    year = result // 100
    month = result % 100

    if not (
        1900 <= year <= 2100
        and 1 <= month <= 12
    ):
        raise ValueError(
            f"Invalid YYYYMM: {value!r}"
        )

    return result


def month_number(value):
    """
    Convert YYYYMM into a continuous month index.
    """
    return (value // 100) * 12 + value % 100


def month_label(value):
    """
    Convert YYYYMM to YYYY-MM.
    """
    return f"{value // 100:04d}-{value % 100:02d}"


def sql_path(path):
    return str(path).replace("'", "''")


def find_zip(year):
    """
    Locate the original Freddie Mac sample ZIP.
    """

    candidates = [
        p for p in RAW.rglob("*.zip")
        if p.name.lower() == f"sample_{year}.zip"
    ]

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one sample_{year}.zip "
            f"under {RAW}; found {len(candidates)}"
        )

    return candidates[0]


# ============================================================
# Load Eligible Loan Events
# ============================================================

def load_targets():

    for p in [EVENTS, HORIZON, REFERENCE]:

        if not p.is_file():
            raise FileNotFoundError(p)

    con = duckdb.connect()

    con.execute("SET threads=2")
    con.execute("SET memory_limit='1GB'")

    query = f"""
        WITH eligible AS (

            SELECT DISTINCT loan_id

            FROM read_parquet('{sql_path(HORIZON)}')

            WHERE TRY_CAST(horizon AS INTEGER) = 12
              AND eligibility = 'Eligible'

        )

        SELECT
            e.loan_id,
            e.vintage,
            e.first_payment,
            e.last_period,
            e.zero_effective,
            e.first_90_period

        FROM read_parquet('{sql_path(EVENTS)}') AS e

        INNER JOIN eligible AS c
            USING (loan_id)

        WHERE TRY_CAST(e.vintage AS INTEGER)
              BETWEEN 2017 AND 2019
    """

    try:

        df = con.execute(query).fetchdf()

    finally:

        con.close()

    targets = {}
    invalid = 0

    for r in df.itertuples(index=False):

        event = ym(r.first_90_period)

        if event is None:
            continue

        if not START_EVENT <= event <= END_EVENT:
            continue

        entry = ym(r.first_payment)
        last = ym(r.last_period)
        zero = ym(r.zero_effective)

        if entry is None or last is None:

            invalid += 1
            continue

        valid = (
            month_number(entry)
            <= month_number(event)
            <= month_number(last)
        )

        if zero is not None:

            valid = (
                valid
                and month_number(event)
                <= month_number(zero)
            )

        if not valid:

            invalid += 1
            continue

        loan_id = str(r.loan_id).strip()

        if loan_id in targets:

            raise ValueError(
                "Duplicate Loan ID in eligible cohort"
            )

        targets[loan_id] = {

            "vintage": int(r.vintage),

            "event": event,
        }

    if invalid:

        raise ValueError(
            f"{invalid} event dates failed "
            "the Task 07 observation-window rules"
        )

    print(
        f"Target first-90 events: {len(targets):,}"
    )

    return targets


# ============================================================
# Reconcile with Task 07
# ============================================================

def reconcile_targets(targets):

    ref = pd.read_csv(REFERENCE)

    ref = ref.loc[
        ref["vintage"].isin(VINTAGES)
        & ref["month"].between(
            "2019-01",
            "2021-12",
        )
    ]

    expected = Counter()

    for r in ref.itertuples(index=False):

        key = (
            int(r.vintage),
            int(str(r.month).replace("-", "")),
        )

        expected[key] += int(r.first90_events)

    observed = Counter(
        (
            item["vintage"],
            item["event"],
        )
        for item in targets.values()
    )

    mismatches = [

        (k, expected[k], observed[k])

        for k in set(expected) | set(observed)

        if expected[k] != observed[k]
    ]

    if mismatches:

        raise ValueError(
            "Task 07 event reconciliation failed. "
            f"First mismatches: {mismatches[:5]}"
        )

    print(
        "Task 07 event reconciliation: PASSED"
    )


# ============================================================
# Read Original Performance ZIPs
# ============================================================

def read_performance(targets):

    # Store only four relevant months per target loan.
    records = {
        loan: {}
        for loan in targets
    }

    # Byte lookup avoids decoding every Loan ID.
    byte_targets = {
        loan.encode("ascii"): loan
        for loan in targets
    }

    for year in VINTAGES:

        path = find_zip(year)

        print(f"Reading {path.name} ...")

        with zipfile.ZipFile(path) as z:

            members = [

                name for name in z.namelist()

                if Path(name).name.lower()
                == f"sample_perf_{year}.txt"
            ]

            if len(members) != 1:

                raise ValueError(
                    f"Cannot identify Performance file "
                    f"inside {path.name}"
                )

            with z.open(members[0]) as f:

                checked_layout = False

                for line in f:

                    # The first column is Loan Identifier.
                    loan_bytes = line.partition(b"|")[0]

                    if not checked_layout:

                        ncols = len(
                            line.rstrip(b"\r\n").split(b"|")
                        )

                        if ncols != EXPECTED_COLS:

                            raise ValueError(
                                f"{path.name}: expected "
                                f"{EXPECTED_COLS} columns; "
                                f"found {ncols}"
                            )

                        checked_layout = True

                    loan = byte_targets.get(loan_bytes)

                    if loan is None:
                        continue

                    meta = targets[loan]

                    if meta["vintage"] != year:

                        raise ValueError(
                            "Loan ID appears in unexpected vintage"
                        )

                    parts = (
                        line.rstrip(b"\r\n").split(b"|")
                    )

                    if len(parts) != EXPECTED_COLS:

                        raise ValueError(
                            f"Incorrect column count for {loan}"
                        )

                    period = int(
                        parts[IDX_PERIOD]
                    )

                    delta = (
                        month_number(period)
                        - month_number(meta["event"])
                    )

                    # The event month and prior three months.
                    if not -3 <= delta <= 0:
                        continue

                    if delta in records[loan]:

                        raise ValueError(
                            "Duplicate loan-month record detected"
                        )

                    dpd = (
                        parts[IDX_DPD]
                        .strip()
                        .decode("ascii")
                    )

                    assistance = (
                        parts[IDX_ASSISTANCE]
                        .strip()
                        .decode("ascii")
                        .upper()
                    )

                    disaster = (
                        parts[IDX_DISASTER]
                        .strip()
                        .decode("ascii")
                        .upper()
                    )

                    if assistance not in (
                        "",
                        "F",
                        "R",
                        "T",
                    ):

                        raise ValueError(
                            "Unexpected assistance code: "
                            f"{assistance}"
                        )

                    if disaster not in (
                        "",
                        "Y",
                    ):

                        raise ValueError(
                            "Unexpected disaster code: "
                            f"{disaster}"
                        )

                    records[loan][delta] = {

                        "dpd": dpd,

                        "assistance": assistance,

                        "disaster": disaster,
                    }

        print("  Finished.")

    return records


# ============================================================
# Build Event-Level Audit Flags
# ============================================================

def build_flags(targets, records):

    rows = []

    missing_event = 0
    dpd_mismatch = 0
    incomplete_pre3 = 0

    for loan, meta in targets.items():

        obs = records[loan]

        event_row = obs.get(0)

        if event_row is None:

            missing_event += 1
            continue

        dpd = event_row["dpd"]

        # 90+ DPD corresponds to >= 3 months delinquent.
        if not (
            dpd.isdigit()
            and int(dpd) >= 3
        ):

            dpd_mismatch += 1

        pre = [

            obs[k]

            for k in (-3, -2, -1)

            if k in obs
        ]

        window = pre + [event_row]

        complete = all(
            k in obs
            for k in (-3, -2, -1)
        )

        if not complete:

            incomplete_pre3 += 1

        def contains_f(items):

            return any(
                x["assistance"] == "F"
                for x in items
            )

        def contains_y(items):

            return any(
                x["disaster"] == "Y"
                for x in items
            )

        f_event = contains_f(
            [event_row]
        )

        f_prior3 = contains_f(
            pre
        )

        f_window = contains_f(
            window
        )

        y_event = contains_y(
            [event_row]
        )

        y_window = contains_y(
            window
        )

        event = meta["event"]

        rows.append({

            "vintage":
                meta["vintage"],

            "year":
                event // 100,

            "event_month":
                month_label(event),

            "f_event":
                f_event,

            "f_prior3":
                f_prior3,

            "f_window":
                f_window,

            "disaster_event":
                y_event,

            "disaster_window":
                y_window,

            "either_window":
                f_window or y_window,

            "other_assistance_event":
                event_row["assistance"] in ("R", "T"),

            "pre3_complete":
                complete,

            "f_window_complete":
                complete and f_window,

            "either_window_complete":
                complete and (f_window or y_window),
        })

    # Event-month records must match the event table.
    if missing_event or dpd_mismatch:

        raise ValueError(
            "Raw Performance validation failed: "
            f"missing event rows={missing_event}, "
            f"DPD mismatches={dpd_mismatch}"
        )

    print(
        "Complete 3-month pre-event histories: "
        f"{len(rows) - incomplete_pre3:,} "
        f"/ {len(rows):,}"
    )

    return (
        pd.DataFrame(rows),
        incomplete_pre3,
    )


# ============================================================
# Aggregations
# ============================================================

def aggregate(df, groups):

    agg = {

        "n_events": (
            "f_event",
            "size",
        ),
    }

    for metric in METRICS:

        agg[f"n_{metric}"] = (
            metric,
            "sum",
        )

    result = (
        df.groupby(
            groups,
            as_index=False,
        )
        .agg(**agg)
    )

    # Share with a forbearance marker in the event month.
    result["pct_f_event"] = (

        100
        * result["n_f_event"]
        / result["n_events"]
    )

    # Share with disaster marker in the event month.
    result["pct_disaster_event"] = (

        100
        * result["n_disaster_event"]
        / result["n_events"]
    )

    # For the four-month window, only complete histories
    # are used in the denominator.
    denominator = (
        result["n_pre3_complete"]
        .replace(0, np.nan)
    )

    result["pct_f_window_complete"] = (

        100
        * result["n_f_window_complete"]
        / denominator
    )

    result["pct_either_window_complete"] = (

        100
        * result["n_either_window_complete"]
        / denominator
    )

    return result


# ============================================================
# Visualization
# ============================================================

def make_figure(df):

    monthly = aggregate(
        df,
        ["event_month"],
    )

    dates = pd.to_datetime(
        monthly["event_month"] + "-01"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    # --------------------------------------------------------
    # Panel 1: First 90+ Event Counts
    # --------------------------------------------------------

    axes[0].bar(
        dates,
        monthly["n_events"],
        width=22,
    )

    axes[0].set(
        title=(
            "First 90+ DPD events: "
            "2017-2019 vintages"
        ),
        ylabel="Number of first-90 events",
    )

    # --------------------------------------------------------
    # Panel 2: Forbearance Overlap
    # --------------------------------------------------------

    # Do not display percentages based on very small
    # monthly event counts.
    adequate = (
        monthly["n_events"] >= 10
    )

    f_event = (
        monthly["pct_f_event"]
        .where(adequate)
    )

    f_window = (
        monthly["pct_f_window_complete"]
        .where(
            adequate
            & (
                monthly["n_pre3_complete"] >= 10
            )
        )
    )

    axes[1].plot(
        dates,
        f_event,
        label="Forbearance in event month",
    )

    axes[1].plot(
        dates,
        f_window,
        label=(
            "Forbearance in event/prior 3M "
            "(complete histories)"
        ),
    )

    axes[1].set(
        ylabel="Share of first-90 events (%)",
        xlabel="Event calendar month",
        ylim=(0, 100),
    )

    for ax in axes:
        ax.grid(alpha=0.2)

    axes[1].legend(
        frameon=False
    )

    fig.suptitle(
        "First 90+ DPD and reported "
        "forbearance, 2019-2021"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "09_forbearance_event_overlap.png",
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Step 1: Load target loans
    # --------------------------------------------------------

    targets = load_targets()

    if not targets:

        raise ValueError(
            "No qualifying first-90 events found"
        )

    # --------------------------------------------------------
    # Step 2: Reconcile with Task 07
    # --------------------------------------------------------

    reconcile_targets(targets)

    # --------------------------------------------------------
    # Step 3: Scan the original ZIP files
    # --------------------------------------------------------

    records = read_performance(
        targets
    )

    # --------------------------------------------------------
    # Step 4: Build flags
    # --------------------------------------------------------

    flags, incomplete = build_flags(
        targets,
        records,
    )

    # --------------------------------------------------------
    # Step 5: Monthly and annual summaries
    # --------------------------------------------------------

    monthly = aggregate(
        flags,
        ["vintage", "event_month"],
    )

    annual = aggregate(
        flags,
        ["vintage", "year"],
    )

    monthly.to_csv(
        OUT / "09_event_flag_monthly_by_vintage.csv",
        index=False,
        float_format="%.4f",
    )

    annual.to_csv(
        OUT / "09_event_flag_year_vintage.csv",
        index=False,
        float_format="%.4f",
    )

    # --------------------------------------------------------
    # Step 6: Reporting audit
    # --------------------------------------------------------

    audit = pd.DataFrame({

        "check": [

            "target_events",

            "validated_event_rows",

            "complete_pre3_histories",

            "incomplete_pre3_histories",
        ],

        "count": [

            len(targets),

            len(flags),

            len(flags) - incomplete,

            incomplete,
        ],
    })

    audit.to_csv(
        OUT / "09_event_reporting_audit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Step 7: Visualization
    # --------------------------------------------------------

    make_figure(flags)

    # --------------------------------------------------------
    # Step 8: Short terminal report
    # --------------------------------------------------------

    print(
        "\n=== Year Summary: All Three Vintages ==="
    )

    summary = aggregate(
        flags,
        ["year"],
    )

    cols = [

        "year",

        "n_events",

        "n_pre3_complete",

        "pct_f_event",

        "pct_f_window_complete",

        "pct_either_window_complete",
    ]

    print(
        summary[cols]
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n=== 2020 Monthly Event Summary ==="
    )

    m2020 = aggregate(

        flags.loc[
            flags["year"] == 2020
        ],

        ["event_month"],
    )

    print(
        m2020[
            [
                "event_month",

                "n_events",

                "n_f_event",

                "n_disaster_event",

                "pct_f_event",

                "pct_f_window_complete",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print(
        "\nSaved 3 CSV files and 1 figure."
    )


if __name__ == "__main__":
    main()