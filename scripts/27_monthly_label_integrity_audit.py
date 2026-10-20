
"""Task 27.3A: Monthly state-transition label integrity audit.

Read-only audit of Freddie Mac 2006–2022 sample performance files.
Run from the credit-portfolio-data project root.
"""

from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import csv

import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "pd_model"

YEARS = range(2006, 2023)
CHUNK = 120_000
LAST_OBSERVED = 202603
MAX_DETAILS_PER_ISSUE = 2_000

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

# Previous Task 26 results: used for reconciliation only.
EXPECTED_STATES = np.array([
    49_740_285,
    544_036,
    167_252,
    603_284,
    53_860,
    614_048,
    12_619,
    7_985,
], dtype=np.int64)

EXPECTED_MATRIX = np.array([
    [48628659, 290097, 1919, 1217, 4, 601700, 97, 5387],
    [203707, 226790, 104174, 713, 3, 5813, 48, 614],
    [21816, 22243, 53303, 67453, 7, 1519, 86, 221],
    [36887, 4657, 7850, 533829, 7151, 4499, 5300, 1696],
    [0, 0, 0, 1, 46694, 0, 7074, 53],
], dtype=np.int64)

DETAIL_FIELDS = [
    "issue",
    "vintage",
    "loan_id",
    "prior_period",
    "current_period",
    "prior_state",
    "current_state",
    "exit_code",
    "effective_exit_period",
    "delta_months",
    "note",
]


# ============================================================
# 1. Date utilities
# ============================================================

def numeric_yyyymm(raw):
    """Parse reporting months into YYYYMM and month ordinals."""

    s = pd.Series(raw, copy=False).astype("string").str.strip()

    numbers = (
        pd.to_numeric(s, errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )

    year = numbers // 100
    month = numbers % 100

    valid = (
        (numbers >= 199001)
        & (numbers <= LAST_OBSERVED)
        & (month >= 1)
        & (month <= 12)
    )

    ordinal = np.where(valid, year * 12 + month, -1)

    return numbers, ordinal, valid


def ordinal_to_yyyymm(ordinal):
    if ordinal < 0:
        return ""

    year, month0 = divmod(int(ordinal) - 1, 12)

    return f"{year:04d}{month0 + 1:02d}"


# ============================================================
# 2. State classification
# Identical classification rules to Task 26
# ============================================================

def classify(dpd_raw, exit_raw):

    n = len(dpd_raw)
    state = np.full(n, -1, dtype=np.int8)

    d = dpd_raw.str.strip().str.upper()
    x = exit_raw.str.strip().str.upper()

    x = x.where(x.eq(""), x.str.zfill(2))

    active = x.eq("").to_numpy()

    for code, s in KNOWN_EXIT.items():
        state[x.eq(code).to_numpy()] = s

    state[active & d.eq("RA").to_numpy()] = 4

    numeric = (
        d.str.fullmatch(r"\d{1,3}")
        .fillna(False)
        .to_numpy()
    )

    dpd = pd.to_numeric(
        d, errors="coerce"
    ).to_numpy(dtype=float)

    use = active & numeric

    state[use & (dpd == 0)] = 0
    state[use & (dpd == 1)] = 1
    state[use & (dpd == 2)] = 2
    state[use & (dpd >= 3)] = 3

    return state, x.to_numpy(), d.to_numpy()


# ============================================================
# 3. Anomaly detail recorder
# ============================================================

class DetailRecorder:

    def __init__(self, writer):
        self.writer = writer
        self.total = Counter()
        self.written = Counter()

    def add(
        self,
        issue,
        vintage,
        loan_id,
        prior_period="",
        current_period="",
        prior_state="",
        current_state="",
        exit_code="",
        effective_exit_period="",
        delta_months="",
        note="",
    ):

        self.total[issue] += 1

        if self.written[issue] >= MAX_DETAILS_PER_ISSUE:
            return

        self.written[issue] += 1

        self.writer.writerow({
            "issue": issue,
            "vintage": vintage,
            "loan_id": loan_id,
            "prior_period": prior_period,
            "current_period": current_period,
            "prior_state": prior_state,
            "current_state": current_state,
            "exit_code": exit_code,
            "effective_exit_period": effective_exit_period,
            "delta_months": delta_months,
            "note": note,
        })


# ============================================================
# 4. One-vintage audit
# ============================================================

def scan_year(year, details):

    zpath = RAW / f"sample_{year}.zip"

    if not zpath.is_file():
        raise FileNotFoundError(zpath)

    qa = Counter()

    state_counts = np.zeros(8, dtype=np.int64)
    matrix = np.zeros((8, 8), dtype=np.int64)

    exit_by_code = defaultdict(Counter)
    offsets_by_code = Counter()

    # Track loan boundaries across chunks.
    closed = set()

    prev_loan = None
    prev_period = -1
    prev_state = -1
    prev_ended = False

    with ZipFile(zpath) as archive:

        members = [
            name
            for name in archive.namelist()
            if Path(name).name.lower()
            == f"sample_perf_{year}.txt"
        ]

        if len(members) != 1:
            raise ValueError(
                f"{zpath}: expected one performance file, "
                f"got {members}"
            )

        with archive.open(members[0]) as stream:

            chunks = pd.read_csv(
                stream,
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
                    "loan",
                    "period",
                    "dpd",
                    "exit",
                    "exit_month",
                ]

                n = len(chunk)

                if not n:
                    continue

                qa["rows"] += n

                loan = (
                    chunk["loan"]
                    .str.strip()
                    .to_numpy()
                )

                periods_text = (
                    chunk["period"]
                    .str.strip()
                    .to_numpy()
                )

                period, ord_month, valid_date = (
                    numeric_yyyymm(chunk["period"])
                )

                state, code, dpd = classify(
                    chunk["dpd"],
                    chunk["exit"],
                )

                eff_text = (
                    chunk["exit_month"]
                    .str.strip()
                    .to_numpy()
                )

                _, eff_ord, eff_valid = numeric_yyyymm(
                    chunk["exit_month"]
                )

                # --------------------------------------------
                # A. Basic validity
                # --------------------------------------------

                qa["invalid_period_rows"] += int(
                    (~valid_date).sum()
                )

                qa["unknown_state_rows"] += int(
                    (state < 0).sum()
                )

                for i in np.flatnonzero(~valid_date):

                    details.add(
                        "invalid_reporting_month",
                        year,
                        loan[i],
                        current_period=periods_text[i],
                        current_state=int(state[i]),
                    )

                for i in np.flatnonzero(state < 0):

                    details.add(
                        "unknown_state",
                        year,
                        loan[i],
                        current_period=periods_text[i],
                        current_state=int(state[i]),
                        exit_code=code[i],
                        note=f"raw DPD={dpd[i]}",
                    )

                # --------------------------------------------
                # B. Exit date integrity
                # --------------------------------------------

                exit_present = np.isin(
                    code,
                    list(KNOWN_EXIT),
                )

                has_eff = (
                    (eff_text != "")
                    & (eff_text != "0")
                    & (eff_text != "000000")
                )

                for i in np.flatnonzero(exit_present):

                    c = str(code[i])
                    stats = exit_by_code[c]

                    stats["exit_rows"] += 1

                    if not has_eff[i]:

                        qa["exit_missing_effective_date"] += 1
                        stats["missing_effective_date"] += 1

                        details.add(
                            "exit_missing_effective_date",
                            year,
                            loan[i],
                            current_period=periods_text[i],
                            current_state=int(state[i]),
                            exit_code=c,
                        )

                    else:

                        # Preserve Task 26's raw-string check.
                        mismatch = (
                            eff_text[i] != periods_text[i]
                        )

                        if mismatch:
                            qa["exit_date_mismatch"] += 1
                            stats["date_mismatch_raw"] += 1

                        if (
                            not eff_valid[i]
                            or not valid_date[i]
                        ):

                            qa["exit_invalid_date_comparison"] += 1
                            stats["invalid_date"] += 1

                            details.add(
                                "exit_invalid_date_comparison",
                                year,
                                loan[i],
                                current_period=periods_text[i],
                                current_state=int(state[i]),
                                exit_code=c,
                                effective_exit_period=eff_text[i],
                            )

                        else:

                            offset = int(
                                eff_ord[i] - ord_month[i]
                            )

                            offsets_by_code[(c, offset)] += 1

                            stats["effective_before_report"] += (
                                offset < 0
                            )

                            stats["effective_same_report"] += (
                                offset == 0
                            )

                            stats["effective_after_report"] += (
                                offset > 0
                            )

                            if mismatch:

                                details.add(
                                    "exit_date_mismatch",
                                    year,
                                    loan[i],
                                    current_period=periods_text[i],
                                    current_state=int(state[i]),
                                    exit_code=c,
                                    effective_exit_period=eff_text[i],
                                    delta_months=offset,
                                    note=(
                                        "delta = effective - "
                                        "reporting (months)"
                                    ),
                                )

                # Effective date without a recognized exit.

                non_exit_with_eff = (
                    (~exit_present) & has_eff
                )

                qa["effective_date_without_known_exit"] += int(
                    non_exit_with_eff.sum()
                )

                for i in np.flatnonzero(non_exit_with_eff):

                    details.add(
                        "effective_date_without_known_exit",
                        year,
                        loan[i],
                        current_period=periods_text[i],
                        current_state=int(state[i]),
                        exit_code=code[i],
                        effective_exit_period=eff_text[i],
                    )

                # --------------------------------------------
                # C. Loan grouping and terminal states
                # --------------------------------------------

                new_groups = np.flatnonzero(
                    np.r_[
                        True,
                        loan[1:] != loan[:-1],
                    ]
                )

                if (
                    prev_loan is not None
                    and loan[0] != prev_loan
                ):
                    closed.add(prev_loan)

                borders = np.r_[new_groups, n]

                post_exit = np.zeros(
                    n, dtype=bool
                )

                ended_at_end = False

                for j, left in enumerate(new_groups):

                    right = int(borders[j + 1])

                    if left > 0:
                        closed.add(loan[left - 1])

                    is_continuation = (
                        j == 0
                        and loan[left] == prev_loan
                    )

                    if (
                        not is_continuation
                        and loan[left] in closed
                    ):
                        raise ValueError(
                            "Non-contiguous loan reappearance: "
                            f"{year} {loan[left]}"
                        )

                    ended = bool(
                        is_continuation and prev_ended
                    )

                    if ended:

                        post_exit[left:right] = True

                    else:

                        terminal_indices = np.flatnonzero(
                            state[left:right] >= 5
                        )

                        if terminal_indices.size:

                            first = (
                                left
                                + int(terminal_indices[0])
                            )

                            post_exit[first + 1:right] = True
                            ended = True

                    if j == len(new_groups) - 1:
                        ended_at_end = ended

                qa["post_exit_rows"] += int(
                    post_exit.sum()
                )

                for i in np.flatnonzero(post_exit):

                    details.add(
                        "post_terminal_record",
                        year,
                        loan[i],
                        current_period=periods_text[i],
                        current_state=int(state[i]),
                        exit_code=code[i],
                        effective_exit_period=eff_text[i],
                    )

                # Same exclusion logic as Task 26.
                state[post_exit] = -1

                state_counts += np.bincount(
                    state[
                        (state >= 0) & valid_date
                    ],
                    minlength=8,
                )

                # --------------------------------------------
                # D. Consecutive monthly transitions
                # --------------------------------------------

                same = np.empty(n, dtype=bool)

                same[0] = loan[0] == prev_loan
                same[1:] = (
                    loan[1:] == loan[:-1]
                )

                last_periods = np.empty(
                    n, dtype=np.int64
                )

                last_periods[0] = prev_period
                last_periods[1:] = ord_month[:-1]

                last_states = np.empty(
                    n, dtype=np.int8
                )

                last_states[0] = prev_state
                last_states[1:] = state[:-1]

                delta = ord_month - last_periods

                both_valid = (
                    same
                    & (ord_month >= 0)
                    & (last_periods >= 0)
                )

                checks = [
                    (
                        "duplicate_month_pair",
                        both_valid & (delta == 0),
                    ),
                    (
                        "backward_month_pair",
                        both_valid & (delta < 0),
                    ),
                    (
                        "gap_month_pair",
                        both_valid & (delta > 1),
                    ),
                ]

                for issue, mask in checks:

                    indices = np.flatnonzero(mask)

                    qa[issue] += len(indices)

                    for i in indices:

                        details.add(
                            issue,
                            year,
                            loan[i],
                            prior_period=ordinal_to_yyyymm(
                                last_periods[i]
                            ),
                            current_period=periods_text[i],
                            prior_state=int(last_states[i]),
                            current_state=int(state[i]),
                            exit_code=code[i],
                            delta_months=int(delta[i]),
                            note="delta = months between rows",
                        )

                # Only valid adjacent active-origin pairs.

                good = (
                    both_valid
                    & (delta == 1)
                    & (last_states >= 0)
                    & (last_states <= 4)
                    & (state >= 0)
                )

                qa["eligible_pairs"] += int(
                    good.sum()
                )

                encoded = (
                    last_states[good].astype(np.int64) * 8
                    + state[good]
                )

                matrix += np.bincount(
                    encoded,
                    minlength=64,
                ).reshape(8, 8)

                # --------------------------------------------
                # E. REO state-transition anomalies
                # --------------------------------------------

                s4_reversal = (
                    good
                    & (last_states == 4)
                    & (state <= 3)
                )

                qa["s4_to_active_reversal"] += int(
                    s4_reversal.sum()
                )

                for i in np.flatnonzero(s4_reversal):

                    details.add(
                        "s4_to_active_reversal",
                        year,
                        loan[i],
                        prior_period=ordinal_to_yyyymm(
                            last_periods[i]
                        ),
                        current_period=periods_text[i],
                        prior_state=int(last_states[i]),
                        current_state=int(state[i]),
                        exit_code=code[i],
                    )

                direct_reo = (
                    good
                    & (last_states <= 2)
                    & (state == 4)
                )

                qa["s0_s2_direct_to_s4"] += int(
                    direct_reo.sum()
                )

                for i in np.flatnonzero(direct_reo):

                    details.add(
                        "s0_s2_direct_to_s4",
                        year,
                        loan[i],
                        prior_period=ordinal_to_yyyymm(
                            last_periods[i]
                        ),
                        current_period=periods_text[i],
                        prior_state=int(last_states[i]),
                        current_state=int(state[i]),
                        exit_code=code[i],
                    )

                # --------------------------------------------
                # F. Terminal transition eligibility
                # --------------------------------------------

                terminal = (
                    (state >= 5)
                    & (state <= 7)
                )

                no_predecessor = (
                    terminal & ~good
                )

                qa[
                    "terminal_without_eligible_previous_month"
                ] += int(no_predecessor.sum())

                for i in np.flatnonzero(no_predecessor):

                    details.add(
                        "terminal_without_eligible_previous_month",
                        year,
                        loan[i],
                        prior_period=(
                            ordinal_to_yyyymm(last_periods[i])
                            if same[i] else ""
                        ),
                        current_period=periods_text[i],
                        prior_state=(
                            int(last_states[i])
                            if same[i] else ""
                        ),
                        current_state=int(state[i]),
                        exit_code=code[i],
                        effective_exit_period=eff_text[i],
                        note=(
                            "terminal lacks a valid adjacent "
                            "active-state parent"
                        ),
                    )

                # --------------------------------------------
                # G. Preserve chunk boundary information
                # --------------------------------------------

                prev_loan = loan[-1]
                prev_period = int(ord_month[-1])
                prev_state = int(state[-1])
                prev_ended = ended_at_end

    return (
        qa,
        state_counts,
        matrix,
        exit_by_code,
        offsets_by_code,
    )


# ============================================================
# 5. Aggregate audit
# ============================================================

def main():

    OUT.mkdir(parents=True, exist_ok=True)

    all_qa = Counter()

    all_states = np.zeros(
        8, dtype=np.int64
    )

    all_matrix = np.zeros(
        (8, 8), dtype=np.int64
    )

    all_exit_stats = defaultdict(Counter)
    all_offsets = Counter()

    per_year = []

    detail_path = OUT / "27_label_anomalies.csv"

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as detail_stream:

        writer = csv.DictWriter(
            detail_stream,
            fieldnames=DETAIL_FIELDS,
        )

        writer.writeheader()

        details = DetailRecorder(writer)

        for year in YEARS:

            (
                qa,
                states,
                matrix,
                exit_stats,
                offsets,
            ) = scan_year(year, details)

            all_qa.update(qa)
            all_states += states
            all_matrix += matrix

            for code, stats in exit_stats.items():
                all_exit_stats[code].update(stats)

            all_offsets.update(offsets)

            per_year.append({
                "vintage": year,
                **qa,
                "s4_to_s3": int(matrix[4, 3]),
            })

            print(
                f"{year}: {qa['rows']:,} rows | "
                f"{qa['eligible_pairs']:,} transitions | "
                f"exit mismatches "
                f"{qa['exit_date_mismatch']:,} | "
                f"month gaps {qa['gap_month_pair']:,}",
                flush=True,
            )

    # --------------------------------------------------------
    # Save audit reports
    # --------------------------------------------------------

    per_year.append({
        "vintage": "ALL",
        **all_qa,
        "s4_to_s3": int(all_matrix[4, 3]),
    })

    pd.DataFrame(per_year).fillna(0).to_csv(
        OUT / "27_label_audit_summary.csv",
        index=False,
    )

    pd.DataFrame(
        all_matrix[:5],
        index=STATES[:5],
        columns=STATES,
    ).to_csv(
        OUT / "27_transition_counts.csv"
    )

    pd.DataFrame({
        "state": STATES,
        "monthly_rows": all_states,
    }).to_csv(
        OUT / "27_state_counts.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "exit_code": code,
            **all_exit_stats[code],
        }
        for code in sorted(all_exit_stats)
    ]).fillna(0).to_csv(
        OUT / "27_exit_timing_by_code.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "exit_code": code,
            "effective_minus_report_months": offset,
            "rows": count,
        }
        for (code, offset), count
        in sorted(all_offsets.items())
    ]).to_csv(
        OUT / "27_exit_date_offsets.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "issue": issue,
            "total_rows": count,
            "saved_detail_rows": details.written[issue],
        }
        for issue, count in sorted(details.total.items())
    ]).to_csv(
        OUT / "27_issue_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Reconcile with Task 26
    # --------------------------------------------------------

    checks = {
        "rows_match_task26": (
            int(all_qa["rows"]) == 51_743_369
        ),
        "eligible_pairs_match_task26": (
            int(all_qa["eligible_pairs"]) == 50_893_281
        ),
        "gaps_match_task26": (
            int(all_qa["gap_month_pair"]) == 88
        ),
        "exit_mismatches_match_task26": (
            int(all_qa["exit_date_mismatch"]) == 810
        ),
        "s4_to_s3_match_task26": (
            int(all_matrix[4, 3]) == 1
        ),
        "all_state_counts_match_task26": (
            np.array_equal(
                all_states,
                EXPECTED_STATES,
            )
        ),
        "full_transition_matrix_match_task26": (
            np.array_equal(
                all_matrix[:5],
                EXPECTED_MATRIX,
            )
        ),
        "no_duplicate_months": (
            all_qa["duplicate_month_pair"] == 0
        ),
        "no_backward_months": (
            all_qa["backward_month_pair"] == 0
        ),
        "no_invalid_reporting_months": (
            all_qa["invalid_period_rows"] == 0
        ),
        "no_unknown_states": (
            all_qa["unknown_state_rows"] == 0
        ),
    }

    lines = [
        "TASK 27.3A - READ-ONLY LABEL INTEGRITY AUDIT",
        "The 2006-2022 samples are not portfolio-vintage representative.",
        "No labels, dates or raw source records were modified.",
        "",
        "TASK26 RECONCILIATION / BASIC INTEGRITY:",
    ]

    lines.extend(
        f"  {'PASS' if passed else 'CHECK'} {name}"
        for name, passed in checks.items()
    )

    lines.extend([
        "",
        "OTHER COUNTS (review, not automatic corrections):",
    ])

    lines.extend(
        f"  {key}: {value:,}"
        for key, value in sorted(all_qa.items())
        if key not in {"rows", "eligible_pairs"}
    )

    lines.extend([
        "",
        "IMPORTANT:",
        "  Timing mismatches are diagnostic; do not silently shift exits.",
        "  S4 -> active transitions require review; do not force to S6.",
        "  A review flag does not automatically imply bad data.",
        "  Feature point-in-time eligibility remains a separate check.",
        "  These reports are diagnostics, not training datasets.",
    ])

    (
        OUT / "27_audit_readme.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Concise terminal output
    # --------------------------------------------------------

    print("\n=== MAIN TOTALS ===")

    for key in [
        "rows",
        "eligible_pairs",
        "gap_month_pair",
        "duplicate_month_pair",
        "backward_month_pair",
        "exit_date_mismatch",
        "exit_missing_effective_date",
        "exit_invalid_date_comparison",
        "effective_date_without_known_exit",
        "post_exit_rows",
        "terminal_without_eligible_previous_month",
        "s4_to_active_reversal",
        "s0_s2_direct_to_s4",
    ]:

        print(
            f"  {key}: {all_qa[key]:,}"
        )

    print(
        f"  s4_to_s3: {all_matrix[4, 3]:,}"
    )

    print("\n=== TASK26 CROSS-CHECK ===")

    for name, passed in checks.items():

        print(
            f"  {'PASS' if passed else 'CHECK'}  {name}"
        )

    print(
        f"\nSaved 8 audit files to: {OUT}"
    )


if __name__ == "__main__":
    main()
