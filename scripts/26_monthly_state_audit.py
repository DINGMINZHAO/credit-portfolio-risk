
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path.cwd()
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs/pd_model"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = range(2006, 2023)
CHUNK = 120_000

STATES = [
    "S0_current",
    "S1_30dpd",
    "S2_60dpd",
    "S3_90plus",
    "S4_REO",
    "S5_payoff",
    "S6_credit_exit",
    "S7_other_exit",
]

KNOWN_EXIT = {
    "01": 5,
    "02": 6,
    "03": 6,
    "09": 6,
    "15": 7,
    "16": 7,
    "96": 7,
}


def classify(dpd_raw, code_raw):
    n = len(dpd_raw)
    state = np.full(n, -1, dtype=np.int8)

    d = dpd_raw.str.strip().str.upper()
    x = code_raw.str.strip().str.upper()
    x = x.where(x.eq(""), x.str.zfill(2))

    active = x.eq("").to_numpy()

    for code, s in KNOWN_EXIT.items():
        state[x.eq(code).to_numpy()] = s

    state[active & d.eq("RA").to_numpy()] = 4

    numeric = d.str.fullmatch(r"\d{1,3}").fillna(False).to_numpy()
    dpd = pd.to_numeric(d, errors="coerce").to_numpy(dtype=float)
    use = active & numeric

    state[use & (dpd == 0)] = 0
    state[use & (dpd == 1)] = 1
    state[use & (dpd == 2)] = 2
    state[use & (dpd >= 3)] = 3

    return state, x.to_numpy(), d.to_numpy()


def scan_year(year):
    zpath = RAW / f"sample_{year}.zip"

    if not zpath.exists():
        raise FileNotFoundError(f"Missing {zpath}")

    counts = np.zeros(8, dtype=np.int64)
    matrix = np.zeros((8, 8), dtype=np.int64)

    qa = Counter()
    exit_codes = Counter()
    invalid_dpd = Counter()

    closed = set()
    prev_loan = None
    prev_period = -1
    prev_state = -1
    prev_ended = False

    with ZipFile(zpath) as z:
        members = [
            m for m in z.namelist()
            if Path(m).name.lower() == f"sample_perf_{year}.txt"
        ]

        if len(members) != 1:
            raise ValueError(
                f"Expected one performance file in {zpath}, got {members}"
            )

        with z.open(members[0]) as f:
            chunks = pd.read_csv(
                f,
                sep="|",
                header=None,
                usecols=[0, 1, 3, 8, 9],
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK,
                encoding="utf-8",
                on_bad_lines="error",
            )

            for chunk in chunks:
                chunk.columns = [
                    "loan", "period", "dpd", "exit", "exit_month"
                ]

                loan = chunk["loan"].str.strip().to_numpy()
                n = len(chunk)

                if not n:
                    continue

                period = pd.to_numeric(
                    chunk["period"].str.strip(),
                    errors="coerce",
                ).fillna(-1).to_numpy(dtype=np.int64)

                yy, mm = period // 100, period % 100

                valid_date = (
                    (yy >= 1990)
                    & (yy <= 2026)
                    & (mm >= 1)
                    & (mm <= 12)
                    & (period <= 202603)
                )

                ord_month = np.where(
                    valid_date, yy * 12 + mm, -1
                )

                state, code, dpd = classify(
                    chunk["dpd"], chunk["exit"]
                )

                qa["rows"] += n
                qa["invalid_period_rows"] += int(
                    (~valid_date).sum()
                )
                qa["unknown_state_rows"] += int(
                    (state < 0).sum()
                )

                for key, count in Counter(
                    code[code != ""]
                ).items():
                    exit_codes[str(key)] += count

                for key, count in Counter(
                    dpd[state < 0]
                ).items():
                    invalid_dpd[str(key)] += count

                exit_present = np.isin(
                    code, list(KNOWN_EXIT)
                )

                eff = chunk["exit_month"].str.strip().to_numpy()
                has_eff = (
                    (eff != "")
                    & (eff != "0")
                    & (eff != "000000")
                )

                qa["exit_date_mismatch"] += int(
                    (
                        exit_present
                        & has_eff
                        & (
                            eff != chunk["period"].str.strip().to_numpy()
                        )
                    ).sum()
                )

                qa["exit_missing_effective_date"] += int(
                    (exit_present & ~has_eff).sum()
                )

                # Check whether loan records are grouped.
                new_group = np.flatnonzero(
                    np.r_[True, loan[1:] != loan[:-1]]
                )

                if prev_loan is not None and loan[0] != prev_loan:
                    closed.add(prev_loan)

                for idx in new_group:
                    if idx == 0 and loan[0] == prev_loan:
                        continue

                    if idx > 0:
                        closed.add(loan[idx - 1])

                    if loan[idx] in closed:
                        raise ValueError(
                            f"Loan ID reappeared non-contiguously "
                            f"in {year}: {loan[idx]}"
                        )

                # Ignore records after a terminal exit.
                post_exit = np.zeros(n, dtype=bool)
                borders = np.r_[new_group, n]
                ended_at_end = False

                for j, left in enumerate(new_group):
                    right = borders[j + 1]

                    ended = bool(
                        j == 0
                        and loan[left] == prev_loan
                        and prev_ended
                    )

                    if ended:
                        post_exit[left:right] = True
                    else:
                        terminal_positions = np.flatnonzero(
                            state[left:right] >= 5
                        )

                        if terminal_positions.size:
                            first = int(terminal_positions[0])
                            post_exit[left + first + 1:right] = True
                            ended = True

                    if j == len(new_group) - 1:
                        ended_at_end = ended

                qa["post_exit_rows"] += int(post_exit.sum())
                state[post_exit] = -1

                counts += np.bincount(
                    state[(state >= 0) & valid_date],
                    minlength=8,
                )

                # Build consecutive-month transitions.
                same = np.empty(n, dtype=bool)
                same[0] = (loan[0] == prev_loan)
                same[1:] = loan[1:] == loan[:-1]

                last_periods = np.empty(n, dtype=np.int64)
                last_periods[0] = prev_period
                last_periods[1:] = ord_month[:-1]

                last_states = np.empty(n, dtype=np.int8)
                last_states[0] = prev_state
                last_states[1:] = state[:-1]

                delta = ord_month - last_periods

                both_valid = (
                    same
                    & (ord_month >= 0)
                    & (last_periods >= 0)
                )

                qa["duplicate_month_pairs"] += int(
                    (both_valid & (delta == 0)).sum()
                )
                qa["backward_month_pairs"] += int(
                    (both_valid & (delta < 0)).sum()
                )
                qa["gap_pairs"] += int(
                    (both_valid & (delta > 1)).sum()
                )

                good = (
                    both_valid
                    & (delta == 1)
                    & (last_states >= 0)
                    & (last_states <= 4)
                    & (state >= 0)
                )

                qa["eligible_pairs"] += int(good.sum())

                qa["unknown_adjacent_pairs"] += int(
                    (
                        both_valid
                        & (delta == 1)
                        & (
                            (last_states < 0)
                            | (state < 0)
                        )
                    ).sum()
                )

                encoded = (
                    last_states[good].astype(np.int64) * 8
                    + state[good]
                )

                matrix += np.bincount(
                    encoded, minlength=64
                ).reshape(8, 8)

                prev_loan = loan[-1]
                prev_period = int(ord_month[-1])
                prev_state = int(state[-1])
                prev_ended = ended_at_end

    if qa["backward_month_pairs"] or qa["duplicate_month_pairs"]:
        raise ValueError(
            f"Unexpected month ordering / duplicates "
            f"in {year}: {dict(qa)}"
        )

    return counts, matrix, qa, exit_codes, invalid_dpd


def main():
    total_counts = np.zeros(8, dtype=np.int64)
    total_matrix = np.zeros((8, 8), dtype=np.int64)

    all_qa = Counter()
    exit_codes = Counter()
    invalid_dpd = Counter()
    year_rows = []

    for year in YEARS:
        counts, matrix, qa, exits, bad_dpd = scan_year(year)

        total_counts += counts
        total_matrix += matrix
        all_qa.update(qa)
        exit_codes.update(exits)
        invalid_dpd.update(bad_dpd)

        year_rows.append({
            "vintage": year,
            **{
                STATES[k]: int(counts[k])
                for k in range(8)
            },
            "valid_transitions": int(matrix.sum()),
            **dict(qa),
        })

        print(
            f"{year}: {qa['rows']:,} monthly rows; "
            f"{matrix.sum():,} usable transitions",
            flush=True,
        )

    state_df = pd.DataFrame({
        "state": STATES,
        "monthly_rows": total_counts,
    })

    count_df = pd.DataFrame(
        total_matrix[:5],
        index=STATES[:5],
        columns=STATES,
    )

    prob_df = (
        count_df.div(
            count_df.sum(axis=1).replace(0, np.nan),
            axis=0,
        ) * 100
    )

    state_df.to_csv(
        OUT / "26_state_counts.csv", index=False
    )
    count_df.to_csv(
        OUT / "26_transition_counts.csv"
    )
    prob_df.round(5).to_csv(
        OUT / "26_transition_probabilities_pct.csv"
    )
    pd.DataFrame(year_rows).fillna(0).to_csv(
        OUT / "26_vintage_quality.csv", index=False
    )
    pd.DataFrame(
        sorted(exit_codes.items()),
        columns=["exit_code", "rows"],
    ).to_csv(
        OUT / "26_exit_code_counts.csv", index=False
    )
    pd.DataFrame(
        sorted(invalid_dpd.items()),
        columns=["raw_dpd_when_state_unknown", "rows"],
    ).to_csv(
        OUT / "26_unknown_dpd.csv", index=False
    )

    print("\n=== STATE COUNTS (MONTHLY ROWS) ===")
    print(state_df.to_string(index=False))

    print("\n=== NEXT-MONTH TRANSITIONS (COUNTS) ===")
    print(count_df.to_string())

    print("\n=== DATA QUALITY ===")
    for k in [
        "rows",
        "eligible_pairs",
        "unknown_state_rows",
        "gap_pairs",
        "duplicate_month_pairs",
        "backward_month_pairs",
        "post_exit_rows",
        "exit_date_mismatch",
        "exit_missing_effective_date",
    ]:
        print(f"{k}: {all_qa[k]:,}")

    print(f"\nSaved 6 CSV reports in {OUT}")


if __name__ == "__main__":
    main()
