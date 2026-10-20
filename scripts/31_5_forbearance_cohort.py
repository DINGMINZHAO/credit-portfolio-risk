
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path.home() / "Desktop/credit-portfolio-data"

SOURCE = (
    ROOT / "outputs/diagnostics/task31_5_assistance"
)

OUTPUT = SOURCE / "cohort_audit"

LAST_MONTH = 202012
HORIZONS = (1, 3, 6)

FIRST_COLS = [
    "loan_id",
    "period_yyyymm",
    "assistance_raw",
]

FOLLOW_COLS = [
    "loan_id",
    "period_yyyymm",
    "assistance_raw",
    "delinquency_raw",
    "zero_balance_raw",
    "modification_raw",
    "payment_deferral_raw",
    "disaster_raw",
]


def clean(value):
    if value is None:
        return ""
    return str(value).strip().upper()


def month_index(yyyymm):
    year, month = divmod(int(yyyymm), 100)

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {yyyymm}")

    return year * 12 + month - 1


def month_label(index):
    year, month = divmod(index, 12)
    return year * 100 + month + 1


def state_band(value):
    value = clean(value)

    if value == "RA":
        return "S4_REO"

    if value.isdigit():
        number = int(value)

        if number == 0:
            return "S0_CURRENT"
        if number == 1:
            return "S1_30DPD"
        if number == 2:
            return "S2_60DPD"

        return "S3_90PLUS"

    return "UNKNOWN"


def batches(path, columns):
    parquet = pq.ParquetFile(path)

    missing = set(columns) - set(
        parquet.schema_arrow.names
    )

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}"
        )

    yield from parquet.iter_batches(
        batch_size=120_000,
        columns=columns,
    )


def identify_first_f(path):
    """
    Find earliest reported F for each loan.
    Only retain loans whose earliest observed F
    occurred in 2017 or 2020.
    """
    first = {}

    for batch in batches(path, FIRST_COLS):
        ids = batch.column(0).to_pylist()
        months = batch.column(1).to_pylist()
        assistance = batch.column(2).to_pylist()

        for loan, period, flag in zip(
            ids, months, assistance
        ):
            if clean(flag) != "F":
                continue

            period = int(period)

            if loan not in first or period < first[loan]:
                first[loan] = period

    return {
        loan: period
        for loan, period in first.items()
        if period // 100 in (2017, 2020)
    }


def collect_followups(path, first):
    """
    Collect only months t-1 through t+6
    for loans selected in the first pass.
    """
    start = {
        loan: month_index(period)
        for loan, period in first.items()
    }

    records = {loan: {} for loan in first}
    duplicates = 0

    for batch in batches(path, FOLLOW_COLS):
        ids = batch.column(0).to_pylist()

        positions = [
            i for i, loan in enumerate(ids)
            if loan in start
        ]

        if not positions:
            continue

        other = [
            batch.column(k).to_pylist()
            for k in range(1, len(FOLLOW_COLS))
        ]

        for i in positions:
            loan = ids[i]
            period = other[0][i]

            delta = (
                month_index(period) - start[loan]
            )

            if delta < -1 or delta > 6:
                continue

            if delta in records[loan]:
                duplicates += 1
                continue

            # Assistance, delinquency, zero balance,
            # modification, deferral, disaster.
            records[loan][delta] = tuple(
                clean(column[i])
                for column in other[1:]
            )

    if duplicates:
        raise ValueError(
            f"{path.name}: {duplicates} duplicate "
            "loan-month records"
        )

    return records


def horizon_outcome(observations, horizon):
    """
    If the loan exits before the horizon,
    preserve its actual zero balance code.

    Otherwise classify its state at t+h.
    """
    for delta in range(1, horizon + 1):
        record = observations.get(delta)

        if record is None:
            continue

        zero_balance = record[2]

        if zero_balance:
            return (
                "EXIT_" + zero_balance,
                None,
            )

    last = observations.get(horizon)

    if last is None:
        return "MISSING_FOLLOWUP", None

    return (
        state_band(last[1]),
        last[0] or "BLANK",
    )


def main():
    paths = sorted(
        SOURCE.glob(
            "assistance_vintage_*.parquet"
        )
    )

    vintages = [
        int(path.stem.rsplit("_", 1)[-1])
        for path in paths
    ]

    if vintages != list(range(2006, 2021)):
        raise ValueError(
            "Expected vintages 2006-2020, "
            f"found {vintages}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    qa = Counter()
    onset = Counter()
    outcomes = Counter()

    print(
        "=== TASK 31.5C: FORBEARANCE COHORT AUDIT ===",
        flush=True,
    )

    for path in paths:
        first = identify_first_f(path)

        if not first:
            continue

        observations = collect_followups(
            path, first
        )

        accepted = 0

        for loan, period in first.items():
            year = period // 100
            records = observations[loan]

            baseline = records.get(0)
            previous = records.get(-1)

            qa[(year, "candidate_first_F")] += 1

            if baseline is None or baseline[0] != "F":
                qa[(year, "invalid_baseline")] += 1
                continue

            if previous is None:
                qa[(year, "missing_previous_month")] += 1
                continue

            if previous[0] == "F":
                qa[(year, "previous_month_F")] += 1
                continue

            if baseline[2]:
                qa[(year, "exit_at_onset")] += 1
                continue

            state = state_band(baseline[1])
            disaster = baseline[5] or "BLANK"

            qa[(year, "accepted_new_F")] += 1
            accepted += 1

            onset[
                (year, period, state, disaster)
            ] += 1

            for horizon in HORIZONS:
                end_month = month_label(
                    month_index(period) + horizon
                )

                # Never inspect beyond 2020-12.
                if end_month > LAST_MONTH:
                    continue

                result, assistance = horizon_outcome(
                    records, horizon
                )

                # Overall and baseline-state breakdown.
                for group in ("ALL", state):
                    outcomes[
                        (year, horizon, group, result)
                    ] += 1

                    if assistance is not None:
                        outcomes[
                            (
                                year,
                                horizon,
                                group,
                                "ASSIST_" + assistance,
                            )
                        ] += 1

        print(
            f"{path.stem}: "
            f"{len(first):,} candidates, "
            f"{accepted:,} eligible loans",
            flush=True,
        )

    qa_rows = [
        {
            "year": year,
            "check": check,
            "loans": count,
        }
        for (year, check), count in sorted(
            qa.items()
        )
    ]

    pd.DataFrame(
        qa_rows,
        columns=["year", "check", "loans"],
    ).to_csv(
        OUTPUT / "cohort_qa.csv",
        index=False,
    )

    onset_rows = [
        {
            "year": year,
            "onset_yyyymm": period,
            "onset_state": state,
            "onset_disaster_flag": disaster,
            "loans": count,
        }
        for (
            year, period, state, disaster
        ), count in sorted(onset.items())
    ]

    pd.DataFrame(
        onset_rows,
        columns=[
            "year",
            "onset_yyyymm",
            "onset_state",
            "onset_disaster_flag",
            "loans",
        ],
    ).to_csv(
        OUTPUT / "cohort_onset.csv",
        index=False,
    )

    outcome_rows = [
        {
            "year": year,
            "horizon_months": horizon,
            "onset_state": state,
            "outcome": result,
            "loans": count,
        }
        for (
            year, horizon, state, result
        ), count in sorted(outcomes.items())
    ]

    pd.DataFrame(
        outcome_rows,
        columns=[
            "year",
            "horizon_months",
            "onset_state",
            "outcome",
            "loans",
        ],
    ).to_csv(
        OUTPUT / "cohort_outcomes.csv",
        index=False,
    )

    print("\n=== COHORT QA ===")

    for year in (2017, 2020):
        summary = {
            check: count
            for (
                y, check
            ), count in qa.items()
            if y == year
        }

        print(year, summary)

    print("\n=== FOLLOW-UP OUTCOMES ===")

    for year in (2017, 2020):
        for horizon in HORIZONS:
            result = {
                outcome: count
                for (
                    y, h, group, outcome
                ), count in outcomes.items()
                if (
                    y == year
                    and h == horizon
                    and group == "ALL"
                    and not outcome.startswith("ASSIST_")
                )
            }

            if not result:
                continue

            total = sum(result.values())

            exits = sum(
                count
                for name, count in result.items()
                if name.startswith("EXIT_")
            )

            severe = result.get(
                "S3_90PLUS", 0
            )

            missing = result.get(
                "MISSING_FOLLOWUP", 0
            )

            print(
                f"{year} +{horizon}M | "
                f"eligible={total:,} | "
                f"exits={exits:,} | "
                f"S3_at_horizon={severe:,} | "
                f"missing={missing:,}",
                flush=True,
            )

    print("\n=== OUTPUT FILES ===")

    for name in (
        "cohort_qa.csv",
        "cohort_onset.csv",
        "cohort_outcomes.csv",
    ):
        print(OUTPUT / name)

    print("\nNo original data modified.")
    print("No 2021-2023 OOT observations accessed.")


if __name__ == "__main__":
    main()
