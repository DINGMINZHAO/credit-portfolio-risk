from pathlib import Path
from collections import Counter
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
HORIZON = PROC / "07_horizon_events.parquet"
REFERENCE = OUT / "09_event_flag_monthly_by_vintage.csv"

VINTAGES = (2017, 2018, 2019)
EVENT_YEAR = 2020
FOLLOWUP = 36

EXPECTED_EVENTS = 5582
N_COLS = 35

# Release 47 Performance file: zero-based positions
COL = {
    "loan": 0,
    "period": 1,
    "dpd": 3,
    "modification": 7,
    "zero_code": 8,
    "zero_date": 9,
    "actual_loss": 21,
    "payment_deferral": 24,
    "forbearance": 29,
}

LOSS_CODES = {"02", "03", "09", "15"}

KNOWN_CODES = {
    "01", "02", "03", "09",
    "15", "16", "96"
}


# ============================================================
# 2. Helpers
# ============================================================

def ym(value):

    if value is None or pd.isna(value):
        return None

    s = str(value).strip()

    if s.lower() in {
        "", "nan", "nat", "none",
        "null", "<na>", "0", "000000"
    }:
        return None

    if re.fullmatch(r"\d{6}(?:\.0+)?", s):
        result = int(s[:6])
    else:
        dt = pd.to_datetime(s, errors="coerce")

        if pd.isna(dt):
            raise ValueError(f"Invalid date: {value!r}")

        result = int(dt.strftime("%Y%m"))

    year, month = divmod(result, 100)

    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        raise ValueError(f"Invalid YYYYMM: {result}")

    return result


def month_index(x):
    return (x // 100) * 12 + x % 100


def path_sql(path):
    return str(path).replace("'", "''")


def parse_money(raw):

    s = raw.strip().decode("ascii")

    if s.upper() in {
        "", "NULL", "NAN", "NA"
    }:
        return None

    value = float(s.replace(",", ""))

    if not np.isfinite(value):
        raise ValueError(f"Invalid money value: {s}")

    return value


def find_zip(year):

    matches = [
        p for p in RAW.rglob("*.zip")
        if p.name.lower() == f"sample_{year}.zip"
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one sample_{year}.zip; "
            f"found {len(matches)}"
        )

    return matches[0]


def exit_category(code):

    if code == "01":
        return "voluntary_payoff"

    if code in {"02", "03", "09"}:
        return "credit_disposition"

    if code in {"15", "16", "96"}:
        return "transfer_or_other_exit"

    return "other_exit"


# ============================================================
# 3. Load the same eligible cohort as Task 08
# ============================================================

def load_targets():

    for path in [EVENTS, HORIZON, REFERENCE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    con = duckdb.connect()

    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=2")

    query = f"""
        WITH eligible AS (
            SELECT DISTINCT loan_id
            FROM read_parquet('{path_sql(HORIZON)}')
            WHERE TRY_CAST(horizon AS INTEGER) = 12
              AND eligibility = 'Eligible'
        )
        SELECT
            e.loan_id,
            e.vintage,
            e.first_payment,
            e.first_90_period,
            e.last_period,
            e.zero_effective

        FROM read_parquet('{path_sql(EVENTS)}') e

        INNER JOIN eligible USING (loan_id)

        WHERE TRY_CAST(e.vintage AS INTEGER)
              BETWEEN 2017 AND 2019
    """

    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    targets = {}

    for row in df.itertuples(index=False):

        event = ym(row.first_90_period)

        if event is None or event // 100 != EVENT_YEAR:
            continue

        entry = ym(row.first_payment)
        last = ym(row.last_period)
        zero = ym(row.zero_effective)

        if entry is None or last is None:
            raise ValueError("Missing observation boundary")

        em = month_index(event)

        if not (
            month_index(entry) <= em
            <= month_index(last)
        ):
            raise ValueError("Invalid first-90 event time")

        if zero is not None and em > month_index(zero):
            raise ValueError("First-90 event after exit")

        loan = str(row.loan_id).strip()

        if loan in targets:
            raise ValueError("Duplicate target Loan ID")

        targets[loan] = {
            "vintage": int(row.vintage),
            "event": event,
            "event_index": em,
        }

    if len(targets) != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS:,} events, "
            f"found {len(targets):,}"
        )

    print(f"Target loans: {len(targets):,}")

    return targets


# ============================================================
# 4. Read post-event performance histories
# ============================================================

def read_histories(targets):

    histories = {
        loan: {}
        for loan in targets
    }

    byte_ids = {
        loan.encode("ascii"): loan
        for loan in targets
    }

    for year in VINTAGES:

        path = find_zip(year)
        print(f"Reading {path.name} ...")

        with zipfile.ZipFile(path) as archive:

            matches = [
                name for name in archive.namelist()
                if Path(name).name.lower()
                == f"sample_perf_{year}.txt"
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Performance member missing: {year}"
                )

            with archive.open(matches[0]) as file:

                for raw in file:

                    loan_bytes = raw.partition(b"|")[0]
                    loan = byte_ids.get(loan_bytes)

                    if loan is None:
                        continue

                    if targets[loan]["vintage"] != year:
                        raise ValueError(
                            "Unexpected vintage for Loan ID"
                        )

                    parts = raw.rstrip(b"\r\n").split(b"|")

                    if len(parts) != N_COLS:
                        raise ValueError(
                            "Unexpected Performance column count"
                        )

                    period = ym(
                        parts[COL["period"]].decode("ascii")
                    )

                    if period is None:
                        raise ValueError("Missing reporting period")

                    delta = (
                        month_index(period)
                        - targets[loan]["event_index"]
                    )

                    if not 0 <= delta <= FOLLOWUP:
                        continue

                    if delta in histories[loan]:
                        raise ValueError(
                            "Duplicate monthly observation"
                        )

                    dpd = (
                        parts[COL["dpd"]]
                        .strip().decode("ascii").upper()
                    )

                    assistance = (
                        parts[COL["forbearance"]]
                        .strip().decode("ascii").upper()
                    )

                    modification = (
                        parts[COL["modification"]]
                        .strip().decode("ascii").upper()
                    )

                    deferral = (
                        parts[COL["payment_deferral"]]
                        .strip().decode("ascii").upper()
                    )

                    code = (
                        parts[COL["zero_code"]]
                        .strip().decode("ascii").upper()
                    )

                    if code:
                        code = code.zfill(2)

                    zero = ym(
                        parts[COL["zero_date"]].decode("ascii")
                    )

                    loss = parse_money(
                        parts[COL["actual_loss"]]
                    )

                    if assistance not in {"", "F", "R", "T"}:
                        raise ValueError(
                            f"Unknown assistance code: {assistance}"
                        )

                    if modification not in {"", "Y", "P"}:
                        raise ValueError(
                            f"Unknown modification flag: {modification}"
                        )

                    if deferral not in {"", "C", "P"}:
                        raise ValueError(
                            f"Unknown payment deferral flag: {deferral}"
                        )

                    histories[loan][delta] = {
                        "period": period,
                        "dpd": dpd,
                        "assistance": assistance,
                        "modification": modification,
                        "deferral": deferral,
                        "zero_code": code,
                        "zero_date": zero,
                        "actual_loss": loss,
                    }

        print("  Finished.")

    return histories


# ============================================================
# 5. Reconcile against Task 08
# ============================================================

def reconcile(targets, histories):

    ref = pd.read_csv(REFERENCE)

    ref = ref.loc[
        ref["vintage"].isin(VINTAGES)
        & ref["event_month"].astype(str)
              .str.startswith("2020-")
    ]

    expected = {
        (
            int(r.vintage),
            str(r.event_month)
        ): (
            int(r.n_events),
            int(r.n_f_event)
        )
        for r in ref.itertuples(index=False)
    }

    count_events = Counter()
    count_f = Counter()

    for loan, meta in targets.items():

        event = histories[loan].get(0)

        if event is None:
            raise ValueError(
                "First-90 event month missing"
            )

        dpd = event["dpd"]

        if not (
            dpd.isdigit()
            and int(dpd) >= 3
        ):
            raise ValueError(
                "Event-month 90+ DPD validation failed"
            )

        month = (
            f"{meta['event'] // 100:04d}-"
            f"{meta['event'] % 100:02d}"
        )

        key = (meta["vintage"], month)

        count_events[key] += 1

        if event["assistance"] == "F":
            count_f[key] += 1

    for key in set(expected) | set(count_events):

        actual = (
            count_events[key],
            count_f[key]
        )

        if actual != expected.get(key, (0, 0)):
            raise ValueError(
                f"Task 08 mismatch: {key}, "
                f"expected={expected.get(key)}, "
                f"actual={actual}"
            )

    print("Task 08 reconciliation: PASSED")


# ============================================================
# 6. Derive 36-month outcomes
# ============================================================

def derive_outcomes(targets, histories):

    results = []

    for loan, meta in targets.items():

        records = histories[loan]
        event_row = records[0]

        f_event = event_row["assistance"] == "F"

        # Identify exit records within the 36-month window.
        exit_records = []

        for delta, row in records.items():

            code = row["zero_code"]

            if not code:
                continue

            zero_date = row["zero_date"]

            if zero_date is None:
                raise ValueError(
                    "Zero Balance Code without effective date"
                )

            exit_delta = (
                month_index(zero_date)
                - meta["event_index"]
            )

            if not 0 <= exit_delta <= FOLLOWUP:
                continue

            exit_records.append(
                (exit_delta, code)
            )

        distinct_exits = set(exit_records)

        if len(distinct_exits) > 1:
            raise ValueError(
                "Conflicting 36-month exit records"
            )

        if distinct_exits:
            exit_delta, code = next(
                iter(distinct_exits)
            )
        else:
            exit_delta, code = None, ""

        # Return to current must occur strictly before exit.
        def before_exit(delta):
            return (
                exit_delta is None
                or delta < exit_delta
            )

        current_months = [
            delta
            for delta, row in records.items()
            if (
                delta > 0
                and row["dpd"] == "00"
                and before_exit(delta)
            )
        ]

        first_current = (
            min(current_months)
            if current_months else None
        )

        # Modification and payment deferral after first 90+
        any_modification = any(
            row["modification"] == "Y"
            for delta, row in records.items()
            if delta > 0 and before_exit(delta)
        )

        any_deferral = any(
            row["deferral"] == "C"
            for delta, row in records.items()
            if delta > 0 and before_exit(delta)
        )

        # Actual Loss is a disposal-specific disclosure.
        # Use the last observed non-null value in the horizon.
        loss_entries = [
            (delta, row["actual_loss"])
            for delta, row in records.items()
            if (
                row["actual_loss"] is not None
                and (
                    exit_delta is None
                    or delta >= exit_delta
                )
            )
        ]

        loss_entries.sort()

        actual_loss = (
            loss_entries[-1][1]
            if loss_entries else None
        )

        if code in LOSS_CODES:

            if actual_loss is None:
                loss_status = "not_reported"
            elif actual_loss > 0:
                loss_status = "positive_loss"
            elif actual_loss < 0:
                loss_status = "negative_gain"
            else:
                loss_status = "reported_zero"

        elif actual_loss is not None:
            loss_status = "unexpected_loss_record"

        else:
            loss_status = "not_applicable"

        # Mutually exclusive endpoint status.
        if code:

            endpoint = exit_category(code)

        elif FOLLOWUP not in records:

            endpoint = "incomplete_followup"

        else:

            last_dpd = records[FOLLOWUP]["dpd"]

            if last_dpd == "00":
                endpoint = "active_current"

            elif (
                last_dpd.isdigit()
                and int(last_dpd) in (1, 2)
            ):
                endpoint = "active_30_89"

            elif (
                last_dpd.isdigit()
                and int(last_dpd) >= 3
            ):
                endpoint = "active_90plus"

            else:
                endpoint = "active_unknown"

        # Observation continuity audit.
        terminal = (
            exit_delta
            if exit_delta is not None
            else FOLLOWUP
        )

        missing_months = sum(
            delta not in records
            for delta in range(terminal + 1)
        )

        results.append({

            "loan_id": loan,

            "vintage": meta["vintage"],
            "event_month": meta["event"],

            "f_event": f_event,
            "group": (
                "F at event"
                if f_event
                else "No F at event"
            ),

            "returned_current": first_current is not None,
            "months_to_first_current": first_current,

            "modified_after_event": any_modification,
            "payment_deferral_after_event": any_deferral,

            "exit_within_36m": bool(code),
            "exit_code": code or None,
            "months_to_exit": exit_delta,

            "endpoint_36m": endpoint,

            "actual_loss": actual_loss,
            "loss_status": loss_status,

            "missing_months": missing_months,
            "complete_monthly_history": missing_months == 0,
        })

    return pd.DataFrame(results)


# ============================================================
# 7. Summary tables
# ============================================================

def make_summaries(df):

    summary = (
        df.groupby("group")
        .agg(
            n_loans=("loan_id", "size"),

            n_returned_current=(
                "returned_current", "sum"
            ),

            n_exited=(
                "exit_within_36m", "sum"
            ),

            n_modified=(
                "modified_after_event", "sum"
            ),

            n_deferral=(
                "payment_deferral_after_event", "sum"
            ),

            n_positive_loss=(
                "loss_status",
                lambda s: (s == "positive_loss").sum()
            ),

            n_incomplete=(
                "endpoint_36m",
                lambda s: (s == "incomplete_followup").sum()
            ),

            n_gapped=(
                "complete_monthly_history",
                lambda s: (~s).sum()
            ),
        )
        .reset_index()
    )

    for col in [
        "returned_current",
        "exited",
        "modified",
        "deferral",
        "positive_loss",
        "incomplete",
        "gapped",
    ]:

        summary[f"pct_{col}"] = (
            100
            * summary[f"n_{col}"]
            / summary["n_loans"]
        )

    endpoint = (
        df.groupby(
            ["group", "endpoint_36m"]
        )
        .size()
        .rename("n_loans")
        .reset_index()
    )

    endpoint["pct_within_group"] = (
        100
        * endpoint["n_loans"]
        / endpoint.groupby("group")["n_loans"]
                 .transform("sum")
    )

    losses = (
        df.groupby(
            ["group", "exit_code", "loss_status"],
            dropna=False
        )
        .agg(
            n_loans=("loan_id", "size"),
            total_reported_amount=(
                "actual_loss", "sum"
            ),
            n_amount_reported=(
                "actual_loss", "count"
            ),
        )
        .reset_index()
    )

    return summary, endpoint, losses


# ============================================================
# 8. Figure: 36-month endpoint composition
# ============================================================

def make_figure(endpoint):

    groups = [
        "F at event",
        "No F at event"
    ]

    categories = [
        "voluntary_payoff",
        "credit_disposition",
        "transfer_or_other_exit",
        "other_exit",
        "active_current",
        "active_30_89",
        "active_90plus",
        "active_unknown",
        "incomplete_followup",
    ]

    matrix = (
        endpoint.pivot_table(
            index="group",
            columns="endpoint_36m",
            values="pct_within_group",
            fill_value=0,
        )
        .reindex(
            index=groups,
            columns=categories,
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    bottom = np.zeros(len(groups))

    for category in categories:

        values = matrix[category].to_numpy()

        if np.all(values == 0):
            continue

        ax.bar(
            groups,
            values,
            bottom=bottom,
            label=category.replace("_", " "),
        )

        bottom += values

    ax.set(
        title=(
            "36-month outcomes after first 90+ DPD "
            "(2020 event cohort)"
        ),
        ylabel="Share of cohort (%)",
        ylim=(0, 100),
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        FIG / "10_post90_36m_outcomes.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. Main
# ============================================================

def main():

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    targets = load_targets()

    histories = read_histories(targets)

    reconcile(targets, histories)

    outcomes = derive_outcomes(
        targets,
        histories,
    )

    if len(outcomes) != EXPECTED_EVENTS:
        raise AssertionError("Cohort size mismatch")

    summary, endpoint, losses = make_summaries(
        outcomes
    )

    # Local loan-level analytical table.
    outcomes.to_parquet(
        PROC / "10_post90_36m_outcomes.parquet",
        index=False,
    )

    summary.to_csv(
        OUT / "10_post90_summary.csv",
        index=False,
        float_format="%.4f",
    )

    endpoint.to_csv(
        OUT / "10_post90_endpoints.csv",
        index=False,
        float_format="%.4f",
    )

    losses.to_csv(
        OUT / "10_post90_loss_coverage.csv",
        index=False,
        float_format="%.4f",
    )

    audit = pd.DataFrame({
        "check": [
            "target_loans",
            "loans_with_monthly_gaps",
            "incomplete_36m_followup",
            "unexpected_loss_records",
            "missing_actual_loss_on_applicable_exit",
            "positive_reported_losses",
        ],

        "count": [
            len(outcomes),

            int(
                (~outcomes["complete_monthly_history"]).sum()
            ),

            int(
                (outcomes["endpoint_36m"]
                 == "incomplete_followup").sum()
            ),

            int(
                (outcomes["loss_status"]
                 == "unexpected_loss_record").sum()
            ),

            int(
                (outcomes["loss_status"]
                 == "not_reported").sum()
            ),

            int(
                (outcomes["loss_status"]
                 == "positive_loss").sum()
            ),
        ],
    })

    audit.to_csv(
        OUT / "10_post90_audit.csv",
        index=False,
    )

    make_figure(endpoint)

    print("\n=== 36-Month Outcome Summary ===")

    columns = [
        "group",
        "n_loans",
        "pct_returned_current",
        "pct_exited",
        "pct_positive_loss",
        "pct_incomplete",
        "pct_gapped",
    ]

    print(
        summary[columns]
        .round(2)
        .to_string(index=False)
    )

    print("\n=== Endpoint Composition ===")

    print(
        endpoint[
            [
                "group",
                "endpoint_36m",
                "n_loans",
                "pct_within_group",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\n=== Data Quality Audit ===")

    print(
        audit.to_string(index=False)
    )

    print(
        "\nSaved 1 Parquet, 4 CSV files, "
        "and 1 figure."
    )


if __name__ == "__main__":
    main()