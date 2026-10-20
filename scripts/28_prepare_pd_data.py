
"""Task 28: Model-ready Freddie Mac monthly transitions.

DATA PREPARATION ONLY. NO MODEL TRAINING.
"""

from collections import Counter
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. Paths and configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LOANS = ROOT / "data/processed/monthly_model_base"

MACRO = (
    ROOT
    / "data/processed/macro"
    / "27e_state_macro_asof_monthly.csv"
)

OUT = (
    ROOT
    / "data/processed/model_ready"
    / "28_monthly_transitions"
)

TMP = OUT.parent / (OUT.name + "_BUILDING")

BATCH_SIZE = 50_000

# Task 27 audited totals
EXPECTED_ROWS = 51_743_369
EXPECTED_VALID = 50_892_470


# ============================================================
# 2. Temporal split
# ============================================================

SPLITS = {
    "early_train": (200601, 201412),
    "validation": (201501, 201712),
    "late_development": (201801, 202012),
    "oot_test": (202101, 202312),
}


# ============================================================
# 3. Feature definitions
# ============================================================

REQUIRED = [
    "loan_id",
    "period_yyyymm",
    "target_period_yyyymm",
    "state_t",
    "next_state_t1",
    "label_eligible",
    "orig_property_state",
    "prior_observed_state",
    "ever_90plus_through_t",
]

NUMERIC = [
    "orig_credit_score",
    "orig_ltv",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_interest_rate_pct",
    "orig_term_months",
    "upb_t",
    "loan_age_reported_t",
    "months_remaining_reported_t",
    "interest_rate_pct_t",
    "months_90plus_observed_through_t",
    "months_delinquent_observed_through_t",
    "ever_90plus_through_t",
]

CATEGORICAL = [
    "orig_property_state",
    "prior_observed_state",
    "orig_occupancy",
    "orig_loan_purpose",
    "modification_flag_t",
]

MACRO_COLS = [
    "hpi_yoy_pct",
    "unemployment_pct",
    "unemployment_change_3m_pp",
    "mortgage30_pct",
]

# Replace known implausible values and sentinels with NaN.
# Actual imputation belongs to the training pipeline.

LIMITS = {
    "orig_credit_score": (300, 850),
    "orig_ltv": (0, 250),
    "orig_cltv": (0, 250),
    "orig_dti": (0, 100),
    "orig_upb": (0, 1e9),
    "orig_interest_rate_pct": (0, 30),
    "orig_term_months": (1, 600),
    "upb_t": (0, 1e9),
    "interest_rate_pct_t": (0, 30),
    "loan_age_reported_t": (0, 600),
    "months_remaining_reported_t": (0, 600),
    "months_90plus_observed_through_t": (0, 600),
    "months_delinquent_observed_through_t": (0, 600),
    "ever_90plus_through_t": (0, 1),
}


# ============================================================
# 4. Utilities
# ============================================================

def normalize_state(series):

    s = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    number = s.str.extract(
        r"^(?:S)?([0-7])(?:\.0)?(?:$|[_\s-])",
        expand=False,
    )

    return pd.to_numeric(
        number,
        errors="coerce",
    ).astype("Int8")


def macro_lookup():

    if not MACRO.exists():
        raise FileNotFoundError(MACRO)

    m = pd.read_csv(
        MACRO,
        usecols=[
            "state",
            "asof_yyyymm",
            *MACRO_COLS,
        ],
    )

    m["state"] = (
        m["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    m["asof_yyyymm"] = (
        pd.to_numeric(m["asof_yyyymm"])
        .astype("int32")
    )

    if (
        len(m) != 12_393
        or m.duplicated(
            ["state", "asof_yyyymm"]
        ).any()
    ):
        raise ValueError(
            "Unexpected macro coverage or duplicate keys"
        )

    for col in MACRO_COLS:

        m[col] = pd.to_numeric(
            m[col],
            errors="coerce",
        ).astype("float32")

    return m


# ============================================================
# 5. Prepare one batch
# ============================================================

def build_batch(raw, vintage, macro, available):

    valid = pd.to_numeric(
        raw["label_eligible"],
        errors="coerce",
    ).eq(1)

    eligible_count = int(valid.sum())

    period = pd.to_numeric(
        raw["period_yyyymm"],
        errors="coerce",
    )

    state = normalize_state(raw["state_t"])

    target = normalize_state(
        raw["next_state_t1"]
    )

    keep = (
        valid
        & period.between(200601, 202312)
        & state.between(0, 4).fillna(False)
        & target.between(0, 7).fillna(False)
    ).fillna(False)

    if not keep.any():
        return None, eligible_count, 0

    x = raw.loc[keep].reset_index(drop=True)

    df = pd.DataFrame(index=x.index)

    # Identifiers and labels
    df["loan_id"] = x["loan_id"].astype("string")

    df["orig_vintage"] = np.int16(vintage)

    df["period_yyyymm"] = pd.to_numeric(
        x["period_yyyymm"]
    ).astype("int32")

    df["target_period_yyyymm"] = pd.to_numeric(
        x["target_period_yyyymm"]
    ).astype("int32")

    df["state_t"] = normalize_state(
        x["state_t"]
    ).astype("int8")

    df["next_state_t1"] = normalize_state(
        x["next_state_t1"]
    ).astype("int8")

    # Check that the destination is exactly next month.
    year = df["period_yyyymm"] // 100
    month = df["period_yyyymm"] % 100

    expected_next = np.where(
        month == 12,
        (year + 1) * 100 + 1,
        year * 100 + month + 1,
    )

    if (
        df["target_period_yyyymm"].to_numpy()
        != expected_next
    ).any():

        raise ValueError(
            f"Nonconsecutive valid label in {vintage}"
        )

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    for col in NUMERIC:

        if col in available:

            values = pd.to_numeric(
                x[col],
                errors="coerce",
            ).astype("float32")

        else:

            values = pd.Series(
                np.nan,
                index=x.index,
                dtype="float32",
            )

        if col in LIMITS:

            lo, hi = LIMITS[col]

            values = values.where(
                values.between(lo, hi)
            )

        df[col] = values

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for col in CATEGORICAL:

        if col in available:

            s = (
                x[col]
                .astype("string")
                .str.strip()
                .str.upper()
            )

            df[col] = s.replace("", pd.NA)

        else:

            df[col] = pd.Series(
                pd.NA,
                index=x.index,
                dtype="string",
            )

    # --------------------------------------------------------
    # Macro as-of join
    # --------------------------------------------------------

    df = df.merge(
        macro,
        left_on=[
            "orig_property_state",
            "period_yyyymm",
        ],
        right_on=[
            "state",
            "asof_yyyymm",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )

    unmatched = int(
        df["_merge"].ne("both").sum()
    )

    df = df.drop(
        columns=[
            "state",
            "asof_yyyymm",
            "_merge",
        ]
    )

    # --------------------------------------------------------
    # Observation-time derived features
    # --------------------------------------------------------

    df["log_orig_upb"] = np.log1p(
        df["orig_upb"]
    ).astype("float32")

    df["log_upb_t"] = np.log1p(
        df["upb_t"]
    ).astype("float32")

    # Positive: existing loan rate exceeds
    # the current market mortgage rate.
    df["refinance_incentive_pp"] = (
        df["interest_rate_pct_t"]
        - df["mortgage30_pct"]
    ).astype("float32")

    # --------------------------------------------------------
    # First 90+ DPD target
    # --------------------------------------------------------

    df["first90_risk_set"] = (
        df["state_t"].isin([0, 1, 2])
        & df["ever_90plus_through_t"]
        .eq(0)
        .fillna(False)
    ).astype("int8")

    df["first90_next"] = (
        df["first90_risk_set"].eq(1)
        & df["next_state_t1"].eq(3)
    ).astype("int8")

    # IMPORTANT:
    # first90_next is only a valid outcome
    # when first90_risk_set == 1.

    return df, eligible_count, unmatched


# ============================================================
# 6. Main pipeline
# ============================================================

def main():

    files = sorted(
        LOANS.glob(
            "27_monthly_base_*.parquet"
        )
    )

    if len(files) != 17:

        raise FileNotFoundError(
            f"Expected 17 Parquet files, found {len(files)}"
        )

    if OUT.exists():

        raise FileExistsError(
            f"Output already exists: {OUT}"
        )

    if TMP.exists():
        shutil.rmtree(TMP)

    TMP.mkdir(parents=True)

    macro = macro_lookup()

    totals = Counter()
    split_totals = Counter()
    missing_optional = set()

    try:

        for index, file in enumerate(files, 1):

            pf = pq.ParquetFile(file)

            available = set(
                pf.schema_arrow.names
            )

            missing = set(REQUIRED) - available

            if missing:

                raise ValueError(
                    f"{file.name}: missing required "
                    f"columns {sorted(missing)}"
                )

            vintage = int(
                file.stem.split("_")[-1]
            )

            missing_optional.update(
                (
                    set(NUMERIC)
                    | set(CATEGORICAL)
                ) - available
            )

            selected = sorted(
                set(
                    REQUIRED
                    + NUMERIC
                    + CATEGORICAL
                ) & available
            )

            writers = {}
            written = 0

            try:

                for batch in pf.iter_batches(
                    batch_size=BATCH_SIZE,
                    columns=selected,
                ):

                    raw = batch.to_pandas()

                    totals["raw"] += len(raw)

                    prepared, valid, unmatched = (
                        build_batch(
                            raw,
                            vintage,
                            macro,
                            available,
                        )
                    )

                    totals["valid"] += valid
                    totals["unmatched"] += unmatched

                    if prepared is None:
                        continue

                    for name, (start, end) in SPLITS.items():

                        select = prepared[
                            "period_yyyymm"
                        ].between(start, end)

                        if not select.any():
                            continue

                        part = prepared.loc[
                            select
                        ].reset_index(drop=True)

                        count = len(part)

                        written += count

                        split_totals[name] += count
                        totals["written"] += count

                        folder = TMP / name
                        folder.mkdir(exist_ok=True)

                        path = (
                            folder
                            / f"vintage_{vintage}.parquet"
                        )

                        table = pa.Table.from_pandas(
                            part,
                            preserve_index=False,
                        )

                        if name not in writers:

                            writers[name] = pq.ParquetWriter(
                                path,
                                table.schema,
                                compression="zstd",
                                use_dictionary=True,
                            )

                        writers[name].write_table(
                            table
                        )

            finally:

                for writer in writers.values():
                    writer.close()

            print(
                f"{index:02d}/17 | "
                f"vintage {vintage}: "
                f"{written:,} rows",
                flush=True,
            )

        # ----------------------------------------------------
        # 7. Final reconciliation
        # ----------------------------------------------------

        if totals["raw"] != EXPECTED_ROWS:

            raise ValueError(
                "Source row count differs from Task 27: "
                f"{totals['raw']:,}"
            )

        if totals["valid"] != EXPECTED_VALID:

            raise ValueError(
                "Eligible label count differs "
                "from Task 27: "
                f"{totals['valid']:,}"
            )

        if totals["written"] != sum(
            split_totals.values()
        ):

            raise ValueError(
                "Temporal split totals do not reconcile"
            )

        if not totals["written"]:

            raise ValueError(
                "No model-ready observations"
            )

        unmatched_pct = (
            totals["unmatched"]
            / totals["written"]
        )

        if unmatched_pct > 0.05:

            raise ValueError(
                "Unexpected macro join failure: "
                f"{unmatched_pct:.2%}"
            )

        # ----------------------------------------------------
        # 8. Save metadata
        # ----------------------------------------------------

        manifest = {
            "training_performed": False,
            "source_rows": totals["raw"],
            "valid_label_rows_all_periods": totals["valid"],
            "prepared_rows_through_2023": totals["written"],
            "macro_unmatched": totals["unmatched"],
            "splits_by_observation_month": dict(
                split_totals
            ),
            "optional_columns_not_present": sorted(
                missing_optional
            ),
            "feature_columns": (
                NUMERIC
                + CATEGORICAL
                + MACRO_COLS
                + [
                    "state_t",
                    "log_orig_upb",
                    "log_upb_t",
                    "refinance_incentive_pp",
                ]
            ),
            "target_columns": [
                "next_state_t1",
                "first90_next",
            ],
            "first90_next_only_valid_where": (
                "first90_risk_set == 1"
            ),
            "notes": [
                "No sampling, training, imputation or scaling.",
                "Splits use reporting calendar time.",
                "Final development combines early_train, "
                "validation and late_development after "
                "validation decisions.",
                "OOT test remains separate.",
                "Macro inputs are lagged historical proxies.",
                "Historical data revisions may remain.",
                "Source loans use equal vintage sampling.",
                "2024-2026 rows remain in original files.",
            ],
        }

        (
            TMP / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Publish completed dataset only after QA passes.
        TMP.rename(OUT)

        # ----------------------------------------------------
        # 9. Final terminal summary
        # ----------------------------------------------------

        print(
            "\n=== DATA READY / NO MODEL TRAINED ==="
        )

        print(
            f"Raw rows: {totals['raw']:,}"
        )

        print(
            "Eligible transitions (full history): "
            f"{totals['valid']:,}"
        )

        print(
            "Model-ready rows through 2023: "
            f"{totals['written']:,}"
        )

        print(
            f"Unmatched macro: "
            f"{totals['unmatched']:,} "
            f"({unmatched_pct:.2%})"
        )

        for name in SPLITS:

            print(
                f"{name}: {split_totals[name]:,}"
            )

        print(f"\nOutput: {OUT}")

    except Exception:

        print(
            f"Incomplete outputs are in: {TMP}",
            flush=True,
        )

        raise


if __name__ == "__main__":
    main()
