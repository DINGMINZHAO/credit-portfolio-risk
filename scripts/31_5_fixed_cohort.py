
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv

import pyarrow.parquet as pq


ROOT = Path.home() / "Desktop/credit-portfolio-data"
DATA = ROOT / "outputs/diagnostics/task31_5_assistance"
OUT = DATA / "fixed_cohort_audit"

COHORTS = {
    "2020_covid_apr_jun": (202004, 202006),
    "2017_same_season_apr_jun": (201704, 201706),
    "2017_disaster_aug_oct": (201708, 201710),
}

COLS = [
    "loan_id",
    "period_yyyymm",
    "assistance_raw",
    "delinquency_raw",
    "zero_balance_raw",
    "modification_raw",
    "payment_deferral_raw",
    "disaster_raw",
]

HORIZONS = (1, 3, 6)


def clean(x):
    return "" if x is None else str(x).strip().upper()


def month_index(yyyymm):
    year, month = divmod(int(yyyymm), 100)
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {yyyymm}")
    return year * 12 + month - 1


def state(dpd):
    dpd = clean(dpd)

    if dpd == "RA":
        return "S4_REO"

    if dpd.isdigit():
        value = int(dpd)

        if value == 0:
            return "S0_CURRENT"
        if value == 1:
            return "S1_30DPD"
        if value == 2:
            return "S2_60DPD"
        return "S3_90PLUS"

    return "UNKNOWN"


def exit_label(code):
    labels = {
        "01": "EXIT_01_PAYOFF",
        "02": "EXIT_02_THIRD_PARTY_SALE",
        "03": "EXIT_03_SHORT_SALE_CHARGEOFF",
        "09": "EXIT_09_REO_DISPOSITION",
        "15": "EXIT_15_WHOLE_LOAN_SALE",
        "16": "EXIT_16_RPL_SECURITIZATION",
        "96": "EXIT_96_DEFECT",
    }
    return labels.get(code, "EXIT_OTHER_" + code)


def write_csv(path, header, rows):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def identify_cohorts(parquet):
    # First observed F in the available 2014-2020 data.
    earliest = {}

    for batch in parquet.iter_batches(
        batch_size=120_000,
        columns=COLS[:3],
    ):
        ids = batch.column(0).to_pylist()
        periods = batch.column(1).to_pylist()
        flags = batch.column(2).to_pylist()

        for loan, period, flag in zip(
            ids, periods, flags
        ):
            if clean(flag) != "F":
                continue

            period = int(period)

            if (
                loan not in earliest
                or period < earliest[loan]
            ):
                earliest[loan] = period

    selected = {}

    for loan, period in earliest.items():
        for cohort, (start, end) in COHORTS.items():
            if start <= period <= end:
                selected[loan] = (period, cohort)
                break

    return selected


def load_selected_histories(parquet, selected):
    # Read only histories needed for selected loans.
    histories = {loan: {} for loan in selected}

    starts = {
        loan: month_index(period)
        for loan, (period, _) in selected.items()
    }

    for batch in parquet.iter_batches(
        batch_size=120_000,
        columns=COLS,
    ):
        ids = batch.column(0).to_pylist()

        positions = [
            i for i, loan in enumerate(ids)
            if loan in selected
        ]

        if not positions:
            continue

        fields = [
            batch.column(j).to_pylist()
            for j in range(1, len(COLS))
        ]

        for i in positions:
            loan = ids[i]

            h = (
                month_index(fields[0][i])
                - starts[loan]
            )

            if not -1 <= h <= 6:
                continue

            if h in histories[loan]:
                raise ValueError(
                    f"Duplicate loan-month: {loan}, h={h}"
                )

            # Tuple positions:
            # 0 assistance
            # 1 delinquency
            # 2 zero balance code
            # 3 modification
            # 4 payment deferral
            # 5 disaster
            histories[loan][h] = tuple(
                clean(column[i])
                for column in fields[1:]
            )

    return histories


def main():
    paths = sorted(
        DATA.glob("assistance_vintage_*.parquet")
    )

    years = sorted(
        int(p.stem.rsplit("_", 1)[-1])
        for p in paths
    )

    if years != list(range(2006, 2021)):
        raise RuntimeError(
            f"Expected 2006-2020 vintages; got {years}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    qa = Counter()
    metrics = defaultdict(Counter)
    outcomes = Counter()

    for path in paths:
        parquet = pq.ParquetFile(path)

        missing_cols = (
            set(COLS)
            - set(parquet.schema_arrow.names)
        )

        if missing_cols:
            raise ValueError(
                f"{path.name}: missing {missing_cols}"
            )

        selected = identify_cohorts(parquet)

        if not selected:
            continue

        histories = load_selected_histories(
            parquet, selected
        )

        for loan, (_, cohort) in selected.items():
            qa[(cohort, "candidates")] += 1

            records = histories[loan]

            if (
                0 not in records
                or records[0][0] != "F"
            ):
                qa[(cohort, "invalid_onset")] += 1
                continue

            if -1 not in records:
                qa[(cohort, "missing_previous")] += 1
                continue

            if records[-1][0] == "F":
                qa[(cohort, "previous_F")] += 1
                continue

            if records[0][2] or records[-1][2]:
                qa[(cohort, "prior_exit")] += 1
                continue

            # Earliest exit during six-month follow-up.
            exit_at = next(
                (
                    h for h in range(1, 7)
                    if h in records and records[h][2]
                ),
                None,
            )

            required_to = (
                exit_at if exit_at is not None else 6
            )

            # Require continuous observations until exit
            # or until six months have elapsed.
            if any(
                h not in records
                for h in range(1, required_to + 1)
            ):
                qa[(cohort, "monthly_gap")] += 1
                continue

            baseline = state(records[0][1])

            if baseline == "UNKNOWN":
                qa[(cohort, "unknown_baseline")] += 1
                continue

            qa[(cohort, "complete_6m")] += 1

            for h in HORIZONS:
                exited = (
                    exit_at is not None
                    and exit_at <= h
                )

                if exited:
                    outcome = exit_label(
                        records[exit_at][2]
                    )
                else:
                    outcome = state(
                        records[h][1]
                    )

                end = min(h, required_to)

                # Observed 90+ at least once AFTER onset.
                # This is not lifetime first-ever 90+.
                ever_severe = any(
                    state(records[t][1]) == "S3_90PLUS"
                    for t in range(1, end + 1)
                )

                # Actual new modification or deferral
                # events reported by horizon.
                deferral = any(
                    records[t][4] == "Y"
                    for t in range(0, end + 1)
                )

                modification = any(
                    records[t][3] == "Y"
                    for t in range(0, end + 1)
                )

                for group in ("ALL", baseline):
                    key = (cohort, group, h)
                    c = metrics[key]

                    c["n"] += 1
                    c["severe_at_h"] += (
                        outcome == "S3_90PLUS"
                    )
                    c["any_severe_after_onset"] += (
                        ever_severe
                    )
                    c["current_at_h"] += (
                        outcome == "S0_CURRENT"
                    )
                    c["exit_01_payoff"] += (
                        outcome == "EXIT_01_PAYOFF"
                    )
                    c["exit_02_03_09"] += (
                        outcome.startswith(
                            ("EXIT_02_", "EXIT_03_",
                             "EXIT_09_")
                        )
                    )
                    c["exit_15_sale"] += (
                        outcome.startswith("EXIT_15_")
                    )
                    c["all_exits"] += (
                        outcome.startswith("EXIT_")
                    )
                    c["deferral_Y_by_h"] += deferral
                    c["modification_Y_by_h"] += (
                        modification
                    )

                    outcomes[
                        (cohort, group, h, outcome)
                    ] += 1

    # Save cohort quality checks.
    write_csv(
        OUT / "cohort_qa.csv",
        ["cohort", "check", "loans"],
        [
            (cohort, check, count)
            for (cohort, check), count
            in sorted(qa.items())
        ],
    )

    # Save full outcome distribution.
    write_csv(
        OUT / "fixed_outcomes.csv",
        [
            "cohort",
            "onset_state",
            "horizon_m",
            "outcome",
            "loans",
        ],
        [
            (cohort, group, h, outcome, count)
            for (
                cohort, group, h, outcome
            ), count in sorted(outcomes.items())
        ],
    )

    fields = [
        "n",
        "severe_at_h",
        "any_severe_after_onset",
        "current_at_h",
        "exit_01_payoff",
        "exit_02_03_09",
        "exit_15_sale",
        "all_exits",
        "deferral_Y_by_h",
        "modification_Y_by_h",
    ]

    write_csv(
        OUT / "fixed_metrics.csv",
        [
            "cohort", "onset_state", "horizon_m",
            *fields,
        ],
        [
            (
                cohort, group, h,
                *[counter.get(k, 0) for k in fields],
            )
            for (cohort, group, h), counter
            in sorted(metrics.items())
        ],
    )

    print("\n=== FIXED COHORT QUALITY ===")

    for cohort in COHORTS:
        print(
            cohort,
            "| candidates:",
            qa[(cohort, "candidates")],
            "| complete 6M:",
            qa[(cohort, "complete_6m")],
            "| gaps:",
            qa[(cohort, "monthly_gap")],
        )

    print("\n=== FIXED COHORT OUTCOMES ===")

    for cohort in COHORTS:
        for h in HORIZONS:
            c = metrics.get(
                (cohort, "ALL", h)
            )

            if not c:
                continue

            n = c["n"]

            print(
                f"{cohort} +{h}M | "
                f"n={n:,} | "
                f"S3={c['severe_at_h']/n:.1%} | "
                f"S0={c['current_at_h']/n:.1%} | "
                f"payoff01={c['exit_01_payoff']/n:.1%} | "
                f"exit02/03/09="
                f"{c['exit_02_03_09']/n:.1%}"
            )

    print("\nSaved:", OUT)
    print("Original data unchanged.")
    print("No 2021-2023 data accessed.")


if __name__ == "__main__":
    main()
