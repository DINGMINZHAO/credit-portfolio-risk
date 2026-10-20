
from __future__ import annotations

import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(
    os.environ.get(
        "CREDIT_PORT_ROOT",
        str(Path.home() / "Desktop/credit-portfolio-data"),
    )
)
DATA = ROOT / "data/processed/model_ready/28_monthly_transitions"

BATCH = 100_000
SAMPLE_FRACTION = 0.0125
SEED = 42

# Same training cutoff as Task 29
TARGET_BEFORE = 201712

NUMERIC = [
    "orig_credit_score",
    "orig_ltv",
    "orig_dti",
    "log_orig_upb",
    "orig_term_months",
    "loan_age_reported_t",
    "months_remaining_reported_t",
    "log_upb_t",
    "interest_rate_pct_t",
    "months_90plus_observed_through_t",
    "months_delinquent_observed_through_t",
    "hpi_yoy_pct",
    "unemployment_pct",
    "unemployment_change_3m_pp",
    "mortgage30_pct",
    "refinance_incentive_pp",
]

CATEGORICAL = [
    "state_t",
    "prior_observed_state",
    "ever_90plus_through_t",
    "orig_property_state",
    "orig_occupancy",
    "orig_loan_purpose",
    "modification_flag_t",
]

TARGET = "next_state_t1"

COLS = list(
    dict.fromkeys(
        NUMERIC
        + CATEGORICAL
        + [TARGET, "target_period_yyyymm"]
    )
)


def memory_gb():
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw = usage.ru_maxrss

    if platform.system() == "Darwin":
        return raw / 2**30

    return raw / 2**20


def state_codes(series):
    values = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^S", "", regex=True)
    )

    return (
        pd.to_numeric(values, errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int16)
    )


def load_sample():
    rng = np.random.default_rng(SEED)

    samples = []
    eligible = 0
    file_count = 0

    start = time.perf_counter()

    for split in ("early_train", "validation"):
        paths = sorted(
            (DATA / split).glob("*.parquet")
        )

        if not paths:
            raise FileNotFoundError(
                f"No files in {DATA / split}"
            )

        print(
            f"Reading {split}: {len(paths)} files",
            flush=True,
        )

        for path in paths:
            file_count += 1

            pf = pq.ParquetFile(path)

            missing = set(COLS) - set(
                pf.schema_arrow.names
            )

            if missing:
                raise ValueError(
                    f"{path.name}: missing {sorted(missing)}"
                )

            for batch in pf.iter_batches(
                columns=COLS,
                batch_size=BATCH,
            ):
                frame = batch.to_pandas()

                months = pd.to_numeric(
                    frame["target_period_yyyymm"],
                    errors="coerce",
                )

                valid = (
                    months < TARGET_BEFORE
                ).fillna(False).to_numpy(dtype=bool)

                eligible += int(valid.sum())

                selected = (
                    valid
                    & (
                        rng.random(len(frame))
                        < SAMPLE_FRACTION
                    )
                )

                if selected.any():
                    samples.append(
                        frame.loc[
                            selected,
                            NUMERIC + CATEGORICAL + [TARGET],
                        ].copy()
                    )

    sample = pd.concat(
        samples,
        ignore_index=True,
    )

    elapsed = time.perf_counter() - start

    print("\n=== DATA LOADING ===")
    print(f"Files scanned: {file_count}")
    print(f"Eligible rows: {eligible:,}")
    print(f"Pilot sample rows: {len(sample):,}")
    print(f"Read + sampling time: {elapsed:.1f}s")
    print(f"Peak process RAM so far: {memory_gb():.2f} GB")

    return sample, eligible


def prepare_features(frame):
    x = frame[
        NUMERIC + CATEGORICAL
    ].copy()

    for col in NUMERIC:
        x[col] = (
            pd.to_numeric(
                x[col],
                errors="coerce",
            )
            .astype(np.float32)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    for col in CATEGORICAL:
        x[col] = (
            x[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return x


def main():
    print(
        "=== TASK 30: RESOURCE PILOT ===",
        flush=True,
    )

    print(f"Machine: {platform.platform()}")

    train, total_eligible = load_sample()

    labels = state_codes(train[TARGET])

    if ((labels < 0) | (labels > 7)).any():
        raise ValueError("Invalid target states")

    counts = np.bincount(
        labels,
        minlength=8,
    )

    if (counts == 0).any():
        raise ValueError(
            "Pilot sample is missing some target states"
        )

    print(
        f"Target counts S0-S7: {counts.tolist()}"
    )

    x = prepare_features(train)

    del train

    try:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            dtype=np.float32,
            sparse_output=True,
        )
    except TypeError:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            dtype=np.float32,
            sparse=True,
        )

    processor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value=0.0,
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]),
                NUMERIC,
            ),
            (
                "categorical",
                onehot,
                CATEGORICAL,
            ),
        ],
        sparse_threshold=1.0,
    )

    start = time.perf_counter()

    matrix = processor.fit_transform(x)

    encode_time = time.perf_counter() - start

    if sparse.issparse(matrix):
        matrix = matrix.tocsr()

        matrix_bytes = (
            matrix.data.nbytes
            + matrix.indices.nbytes
            + matrix.indptr.nbytes
        )

        format_name = "Sparse CSR"

    else:
        matrix_bytes = matrix.nbytes
        format_name = "Dense"

    estimated_gb = (
        matrix_bytes
        / len(labels)
        * total_eligible
        / 2**30
    )

    print("\n=== FEATURE MATRIX ===")

    print(f"Format: {format_name}")
    print(f"Shape: {matrix.shape}")
    print(
        f"Pilot matrix memory: "
        f"{matrix_bytes / 2**20:.1f} MiB"
    )
    print(
        f"Estimated full matrix memory: "
        f"{estimated_gb:.2f} GiB"
    )
    print(f"Encoding time: {encode_time:.1f}s")
    print(
        f"Peak process RAM so far: "
        f"{memory_gb():.2f} GB"
    )

    del x

    # Short optimizer run for speed measurement only.
    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=25,
        tol=1e-3,
    )

    start = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(matrix, labels)

    fit_time = time.perf_counter() - start

    probabilities = model.predict_proba(
        matrix[:min(20_000, len(labels))]
    )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Invalid model probabilities"
        )

    print("\n=== TRAINING PILOT ===")

    print(f"Training time: {fit_time:.1f}s")
    print(f"Iterations: {model.n_iter_.tolist()}")
    print(
        f"Peak process RAM: "
        f"{memory_gb():.2f} GB"
    )

    nonconverged = any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught
    )

    print(
        "Convergence reached: "
        f"{'No' if nonconverged else 'No warning'}"
    )

    print("\n=== DONE ===")
    print(
        "Performance pilot only. "
        "No validation or test set accessed."
    )
    print(
        "No final model has been trained."
    )


if __name__ == "__main__":
    main()
