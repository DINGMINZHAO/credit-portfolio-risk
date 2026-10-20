
"""Task 27.3B — Freddie Mac monthly loan-state modelling base.

Run from project root. Read-only source ZIPs; one Parquet per vintage.
No macro scenarios or future-horizon outcomes are introduced here.
"""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path.cwd()
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/monthly_model_base"
REPORT = ROOT / "outputs/pd_model"

YEARS = range(2006, 2023)
CHUNK = 120_000

# Zero-based Freddie Mac sample file column positions.
ORIG_POS = [0, 1, 7, 8, 9, 10, 11, 12, 16, 19, 20, 21]

ORIG_NAMES = [
    "orig_credit_score",
    "orig_first_payment_yyyymm",
    "orig_occupancy",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_ltv",
    "orig_interest_rate_pct",
    "orig_property_state",
    "loan_id",
    "orig_loan_purpose",
    "orig_term_months",
]

PERF_POS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]

PERF_NAMES = [
    "loan_id",
    "period_raw",
    "upb_raw",
    "dpd_raw",
    "age_raw",
    "remain_raw",
    "mod_raw",
    "exit_raw",
    "exit_month_raw",
    "rate_raw",
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

REASONS = {
    0: "valid_next_month",
    1: "no_next_record",
    2: "gap_to_next_record",
    3: "ambiguous_exit_date",
    4: "reo_to_active_reversal",
    5: "terminal_origin",
    6: "invalid_month_or_state",
}


# ============================================================
# 1. Dates and state definitions
# ============================================================

def month_ordinal(period):
    """Convert YYYYMM integers to consecutive month numbers."""

    y = period // 100
    m = period % 100

    valid = (
        (y >= 1990)
        & (y <= 2026)
        & (m >= 1)
        & (m <= 12)
        & (period <= 202603)
    )

    return np.where(
        valid, y * 12 + m, -1
    ).astype(np.int32)


def classify(dpd_col, exit_col):
    """Replicate Task 26 state definitions."""

    state = np.full(
        len(dpd_col), -1, dtype=np.int8
    )

    d = dpd_col.str.strip().str.upper()
    x = exit_col.str.strip().str.upper()

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

    state[active & numeric & (dpd == 0)] = 0
    state[active & numeric & (dpd == 1)] = 1
    state[active & numeric & (dpd == 2)] = 2
    state[active & numeric & (dpd >= 3)] = 3

    return state, x


# ============================================================
# 2. Read origination features
# ============================================================

def get_member(archive, year, kind):

    expected = f"sample_{kind}_{year}.txt"

    matches = [
        m for m in archive.namelist()
        if Path(m).name.lower() == expected
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected {expected}, got {matches}"
        )

    return matches[0]


def read_orig(archive, year):

    with archive.open(
        get_member(archive, year, "orig")
    ) as f:

        df = pd.read_csv(
            f,
            sep="|",
            header=None,
            usecols=ORIG_POS,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )

    df.columns = ORIG_NAMES
    df["loan_id"] = df["loan_id"].str.strip()

    if df["loan_id"].duplicated().any():
        raise ValueError(
            f"Duplicate origination Loan ID: {year}"
        )

    if df["loan_id"].eq("").any():
        raise ValueError(
            f"Empty origination Loan ID: {year}"
        )

    numeric_columns = [
        "orig_credit_score",
        "orig_cltv",
        "orig_dti",
        "orig_upb",
        "orig_ltv",
        "orig_interest_rate_pct",
        "orig_term_months",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col].str.strip(),
            errors="coerce",
        ).astype("float32")

    # Freddie Mac missing-value codes.
    df.loc[
        df["orig_credit_score"] == 9999,
        "orig_credit_score",
    ] = np.nan

    for col in ["orig_cltv", "orig_dti", "orig_ltv"]:

        df.loc[
            df[col] == 999, col
        ] = np.nan

    df["orig_first_payment_yyyymm"] = (
        pd.to_numeric(
            df["orig_first_payment_yyyymm"].str.strip(),
            errors="coerce",
        )
        .fillna(-1)
        .astype("int32")
    )

    for col in [
        "orig_occupancy",
        "orig_property_state",
        "orig_loan_purpose",
    ]:
        df[col] = df[col].str.strip()

    return df


# ============================================================
# 3. Keep complete loan histories together
# ============================================================

def complete_groups(chunks):
    """
    A loan may cross a read_csv chunk boundary.
    Never split its history when generating features.
    """

    tail = None
    seen = set()

    for chunk in chunks:

        chunk.columns = PERF_NAMES

        chunk["loan_id"] = (
            chunk["loan_id"].str.strip()
        )

        if tail is None:
            frame = chunk
        else:
            frame = pd.concat(
                [tail, chunk],
                ignore_index=True,
            )

        ids = frame["loan_id"].to_numpy()

        different = np.flatnonzero(
            ids != ids[-1]
        )

        cut = (
            int(different[-1] + 1)
            if len(different)
            else 0
        )

        if cut:

            done = frame.iloc[:cut].copy()
            did = done["loan_id"].to_numpy()

            current_ids = did[
                np.r_[True, did[1:] != did[:-1]]
            ].tolist()

            if len(set(current_ids)) != len(current_ids):
                raise ValueError(
                    "Non-contiguous loan repeated "
                    "within a performance batch"
                )

            if any(i in seen for i in current_ids):
                raise ValueError(
                    "Non-contiguous loan reappearance"
                )

            seen.update(current_ids)

            yield done

        tail = frame.iloc[cut:].copy()

    if tail is not None and not tail.empty:

        current_ids = (
            tail["loan_id"]
            .drop_duplicates()
            .tolist()
        )

        if any(i in seen for i in current_ids):
            raise ValueError(
                "Loan reappeared in final batch"
            )

        yield tail


# ============================================================
# 4. Build monthly observations and transition labels
# ============================================================

def build_batch(raw, orig):

    raw = raw.reset_index(drop=True)

    n = len(raw)
    ids = raw["loan_id"]

    gid = ids.ne(ids.shift()).cumsum()
    grouped = raw.groupby(gid, sort=False)

    month = (
        pd.to_numeric(
            raw["period_raw"].str.strip(),
            errors="coerce",
        )
        .fillna(-1)
        .to_numpy(dtype=np.int32)
    )

    ordinal = month_ordinal(month)

    state, exit_code = classify(
        raw["dpd_raw"],
        raw["exit_raw"],
    )

    exit_month = (
        raw["exit_month_raw"]
        .str.strip()
        .to_numpy()
    )

    # Next-month observations: LABEL information only.
    raw_next_state = (
        pd.Series(state)
        .groupby(gid, sort=False)
        .shift(-1)
        .fillna(-1)
        .to_numpy(dtype=np.int8)
    )

    raw_next_month = (
        pd.Series(month)
        .groupby(gid, sort=False)
        .shift(-1)
        .fillna(-1)
        .to_numpy(dtype=np.int32)
    )

    next_ord = month_ordinal(raw_next_month)
    delta = next_ord - ordinal

    exit_code_next = (
        exit_code
        .groupby(gid, sort=False)
        .shift(-1)
        .fillna("")
        .to_numpy()
    )

    exit_month_next = (
        raw["exit_month_raw"]
        .str.strip()
        .groupby(gid, sort=False)
        .shift(-1)
        .fillna("")
        .to_numpy()
    )

    # Reject transitions to exits with ambiguous timing.
    mismatch_next = (
        np.isin(
            exit_code_next,
            list(KNOWN_EXIT),
        )
        & (
            exit_month_next
            != raw_next_month.astype(str)
        )
    )

    reason = np.zeros(n, dtype=np.int8)

    reason[raw_next_month == -1] = 1

    reason[
        (raw_next_month != -1)
        & (delta != 1)
    ] = 2

    reason[
        mismatch_next
        & (reason == 0)
    ] = 3

    reason[
        (state == 4)
        & (raw_next_state >= 0)
        & (raw_next_state <= 3)
        & (reason == 0)
    ] = 4

    reason[state >= 5] = 5

    reason[
        (state < 0)
        | (ordinal < 0)
        | (
            (raw_next_state < 0)
            & (reason == 0)
        )
    ] = 6

    valid = reason == 0

    next_state = np.where(
        valid, raw_next_state, -1
    ).astype(np.int8)

    target_month = np.where(
        valid, raw_next_month, -1
    ).astype(np.int32)

    # Historical features: information available through t.
    prior_state = (
        pd.Series(state)
        .groupby(gid, sort=False)
        .shift(1)
        .fillna(-1)
        .to_numpy(dtype=np.int8)
    )

    previous_ordinal = (
        pd.Series(ordinal)
        .groupby(gid, sort=False)
        .shift(1)
        .fillna(-1)
        .to_numpy(dtype=np.int32)
    )

    gap_before = (
        (previous_ordinal != -1)
        & (ordinal - previous_ordinal != 1)
    )

    count90 = (
        pd.Series(
            (state == 3).astype("int32")
        )
        .groupby(gid, sort=False)
        .cumsum()
        .to_numpy(dtype=np.int32)
    )

    delinquent_count = (
        pd.Series(
            (
                (state >= 1)
                & (state <= 3)
            ).astype("int32")
        )
        .groupby(gid, sort=False)
        .cumsum()
        .to_numpy(dtype=np.int32)
    )

    gap_count = (
        pd.Series(
            gap_before.astype("int32")
        )
        .groupby(gid, sort=False)
        .cumsum()
        .to_numpy(dtype=np.int32)
    )

    first_ord = (
        pd.Series(ordinal)
        .groupby(gid, sort=False)
        .transform("first")
        .to_numpy(dtype=np.int32)
    )

    months_seen = (
        grouped.cumcount()
        .to_numpy(dtype=np.int16)
        + 1
    )

    out = pd.DataFrame({

        # Identification and labels
        "loan_id": ids.to_numpy(),
        "vintage": np.full(n, -1, dtype=np.int16),
        "period_yyyymm": month,
        "state_t": state,
        "next_state_t1": next_state,
        "target_period_yyyymm": target_month,
        "label_reason": reason,
        "label_eligible": valid,

        # Current-period characteristics
        "upb_t": pd.to_numeric(
            raw["upb_raw"].str.strip(),
            errors="coerce",
        ).astype("float32").to_numpy(),

        "loan_age_reported_t": pd.to_numeric(
            raw["age_raw"].str.strip(),
            errors="coerce",
        ).astype("float32").to_numpy(),

        "months_remaining_reported_t": pd.to_numeric(
            raw["remain_raw"].str.strip(),
            errors="coerce",
        ).astype("float32").to_numpy(),

        "interest_rate_pct_t": pd.to_numeric(
            raw["rate_raw"].str.strip(),
            errors="coerce",
        ).astype("float32").to_numpy(),

        "modification_flag_t": (
            raw["mod_raw"].str.strip().to_numpy()
        ),

        # Historical behavior through current month
        "prior_observed_state": prior_state,
        "history_gap_before_t": gap_before,
        "history_gap_count_through_t": gap_count,
        "observed_months_through_t": months_seen,

        "months_since_first_observation_t": (
            ordinal - first_ord
        ).astype(np.int16),

        "months_90plus_observed_through_t": count90,
        "ever_90plus_through_t": count90 > 0,

        "months_delinquent_observed_through_t": (
            delinquent_count
        ),

        # Audit fields: NEVER automatic model features.
        "exit_code_at_t_AUDIT_ONLY": (
            exit_code.to_numpy()
        ),

        "exit_effective_yyyymm_at_t_AUDIT_ONLY": (
            exit_month
        ),

        "s0_s2_to_reo_next_AUDIT_ONLY": (
            valid
            & (state <= 2)
            & (next_state == 4)
        ),
    })

    # Add origination features without changing label order.
    out = out.merge(
        orig,
        on="loan_id",
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )

    missing = int(
        out["_merge"].ne("both").sum()
    )

    out = out.drop(columns="_merge")

    return out, missing


# ============================================================
# 5. Write one compressed Parquet per vintage
# ============================================================

def build_vintage(year):

    zpath = RAW / f"sample_{year}.zip"

    if not zpath.exists():
        raise FileNotFoundError(zpath)

    output_path = (
        OUT / f"27_monthly_base_{year}.parquet"
    )

    tmp_path = (
        OUT / f"27_monthly_base_{year}.parquet.partial"
    )

    counts = Counter()

    state_counts = np.zeros(
        8, dtype=np.int64
    )

    matrix = np.zeros(
        (8, 8), dtype=np.int64
    )

    raw_matrix = np.zeros(
        (8, 8), dtype=np.int64
    )

    writer = None

    try:

        with ZipFile(zpath) as archive:

            orig = read_orig(archive, year)
            counts["orig_loans"] = len(orig)

            with archive.open(
                get_member(archive, year, "perf")
            ) as f:

                reader = pd.read_csv(
                    f,
                    sep="|",
                    header=None,
                    usecols=PERF_POS,
                    dtype=str,
                    keep_default_na=False,
                    chunksize=CHUNK,
                    encoding="utf-8",
                    on_bad_lines="error",
                )

                for raw in complete_groups(reader):

                    frame, missing = build_batch(
                        raw, orig
                    )

                    frame["vintage"] = np.int16(year)

                    counts["missing_orig_rows"] += missing
                    counts["rows"] += len(frame)

                    counts["loans_seen"] += (
                        frame["loan_id"].nunique()
                    )

                    ok = (
                        frame["label_eligible"].to_numpy()
                    )

                    counts["valid_labels"] += int(
                        ok.sum()
                    )

                    for status, number in (
                        frame["label_reason"]
                        .value_counts()
                        .items()
                    ):

                        counts[
                            f"reason_{int(status)}"
                        ] += int(number)

                    curr = (
                        frame["state_t"].to_numpy()
                    )

                    dest = (
                        frame["next_state_t1"].to_numpy()
                    )

                    state_counts += np.bincount(
                        curr[curr >= 0],
                        minlength=8,
                    )[:8]

                    pairs = (
                        curr[ok].astype(np.int64) * 8
                        + dest[ok]
                    )

                    matrix += np.bincount(
                        pairs,
                        minlength=64,
                    ).reshape(8, 8)

                    # Recover the original Task 26 matrix,
                    # before our new label exclusions.
                    r = frame["label_reason"].to_numpy()

                    raw_ok = (
                        ok
                        | (r == 3)
                        | (r == 4)
                    )

                    raw_pairs = (
                        curr[raw_ok].astype(np.int64) * 8
                    )

                    raw_dest = np.where(
                        r[raw_ok] == 0,
                        dest[raw_ok],
                        -1,
                    )

                    candidate_dest = (
                        frame["state_t"]
                        .groupby(
                            frame["loan_id"],
                            sort=False,
                        )
                        .shift(-1)
                        .fillna(-1)
                        .to_numpy(dtype=np.int8)
                    )

                    raw_dest = np.where(
                        r[raw_ok] == 0,
                        raw_dest,
                        candidate_dest[raw_ok],
                    )

                    raw_matrix += np.bincount(
                        raw_pairs + raw_dest,
                        minlength=64,
                    ).reshape(8, 8)

                    if missing:
                        raise ValueError(
                            f"{year}: {missing} rows "
                            "lack origination matches"
                        )

                    table = pa.Table.from_pandas(
                        frame,
                        preserve_index=False,
                    )

                    if writer is None:

                        writer = pq.ParquetWriter(
                            tmp_path,
                            table.schema,
                            compression="zstd",
                        )

                    else:

                        table = table.cast(
                            writer.schema
                        )

                    writer.write_table(table)

        if writer is None:
            raise ValueError(
                f"No performance records: {year}"
            )

        writer.close()
        writer = None

        if counts["loans_seen"] != counts["orig_loans"]:
            raise ValueError(
                f"{year}: loan count mismatch"
            )

        tmp_path.replace(output_path)

    finally:

        if writer is not None:
            writer.close()

    return counts, state_counts, matrix, raw_matrix


# ============================================================
# 6. Aggregate QA and reconciliation
# ============================================================

def main():

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    rows = []

    all_states = np.zeros(
        8, dtype=np.int64
    )

    all_matrix = np.zeros(
        (8, 8), dtype=np.int64
    )

    raw_matrix = np.zeros(
        (8, 8), dtype=np.int64
    )

    for year in YEARS:

        stats, states, matrix, raw = build_vintage(
            year
        )

        all_states += states
        all_matrix += matrix
        raw_matrix += raw

        rows.append({
            "vintage": year,
            **dict(stats),
        })

        print(
            f"{year}: "
            f"{stats['rows']:,} rows | "
            f"{stats['valid_labels']:,} valid labels | "
            f"{stats['reason_3']:,} timing exclusions | "
            f"{stats['reason_4']:,} REO reversals",
            flush=True,
        )

    summary = pd.DataFrame(
        rows
    ).fillna(0)

    summary.to_csv(
        REPORT / "27b_monthly_base_qa.csv",
        index=False,
    )

    pd.DataFrame(
        all_matrix[:5],
        index=range(5),
        columns=range(8),
    ).to_csv(
        REPORT / "27b_transition_counts.csv"
    )

    pd.DataFrame({
        "state": range(8),
        "count": all_states,
    }).to_csv(
        REPORT / "27b_state_counts.csv",
        index=False,
    )

    print("\n=== TOTALS ===")

    keys = [
        "rows",
        "valid_labels",
        "reason_1",
        "reason_2",
        "reason_3",
        "reason_4",
        "reason_5",
        "reason_6",
        "missing_orig_rows",
    ]

    for k in keys:

        value = (
            int(summary[k].sum())
            if k in summary
            else 0
        )

        print(f"{k}: {value:,}")

    print("\n=== TASK 26 RECONCILIATION ===")

    checks = {
        "total_rows": (
            int(summary["rows"].sum())
            == 51_743_369
        ),
        "original_eligible_pairs": (
            int(raw_matrix.sum())
            == 50_893_281
        ),
        "monthly_gaps": (
            int(summary["reason_2"].sum())
            == 88
        ),
        "S4_to_S3_in_raw": (
            raw_matrix[4, 3] == 1
        ),
    }

    for k, passed in checks.items():
        print(
            f"{k}: {'PASS' if passed else 'CHECK'}"
        )

    print("\n=== LABEL STATUS DEFINITIONS ===")

    for code, description in REASONS.items():
        print(f"{code}: {description}")

    print(
        "\nSaved monthly Parquets in "
        "data/processed/monthly_model_base/"
    )


if __name__ == "__main__":
    main()
