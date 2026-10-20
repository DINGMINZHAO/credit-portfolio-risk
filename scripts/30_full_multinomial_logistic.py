
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax


ROOT = Path(
    os.getenv(
        "CREDIT_PORT_ROOT",
        str(Path.home() / "Desktop/credit-portfolio-data"),
    )
)

DATA = ROOT / "data/processed/model_ready/28_monthly_transitions"
OUT = ROOT / "outputs/modeling/task30_streaming_logistic"
BASELINE = ROOT / "outputs/modeling/task29_benchmark/validation_summary.csv"

BATCH_SIZE = 75_000
L2 = 1e-6
MAX_ITER = 55

# Same conservative boundaries as Task 29
TRAIN_TARGET_BEFORE = 201712
VALID_TARGET_BEFORE = 202012

NUM = [
    "orig_credit_score",
    "orig_ltv",
    "orig_dti",
    "log_orig_upb",
    "orig_term_months",
    "loan_age_reported_t",
    "months_remaining_reported_t",
    "log_upb_t",
    "months_90plus_observed_through_t",
    "months_delinquent_observed_through_t",
    "hpi_yoy_pct",
    "unemployment_pct",
    "unemployment_change_3m_pp",
    "mortgage30_pct",
    "refinance_incentive_pp",
]

CAT = [
    "state_t",
    "prior_observed_state",
    "ever_90plus_through_t",
    "orig_property_state",
    "orig_occupancy",
    "orig_loan_purpose",
    "modification_flag_t",
]

META = [
    "period_yyyymm",
    "target_period_yyyymm",
    "next_state_t1",
    "first90_risk_set",
    "first90_next",
]

READ = list(dict.fromkeys(NUM + CAT + META))


def state_codes(s):
    text = (
        s.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^S", "", regex=True)
    )
    return (
        pd.to_numeric(text, errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int16)
    )


def binary_codes(s):
    v = (
        s.astype("string")
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0"})
    )
    return (
        pd.to_numeric(v, errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int8)
    )


def cat_strings(s):
    return (
        s.astype("string")
        .fillna("__MISSING__")
        .str.strip()
        .replace("", "__MISSING__")
        .astype(str)
    )


def batches(split):
    paths = sorted((DATA / split).glob("*.parquet"))

    if not paths:
        raise FileNotFoundError(
            f"No parquet in {DATA / split}"
        )

    for path in paths:
        pf = pq.ParquetFile(path)

        missing = sorted(
            set(READ) - set(pf.schema_arrow.names)
        )
        if missing:
            raise ValueError(
                f"{path}: missing {missing}"
            )

        for chunk in pf.iter_batches(
            batch_size=BATCH_SIZE,
            columns=READ,
        ):
            yield chunk.to_pandas()


def select(frame, split):
    p = (
        pd.to_numeric(
            frame["period_yyyymm"], errors="coerce"
        )
        .fillna(-1)
        .to_numpy(dtype=np.int32)
    )

    tp = (
        pd.to_numeric(
            frame["target_period_yyyymm"],
            errors="coerce",
        )
        .fillna(-1)
        .to_numpy(dtype=np.int32)
    )

    if split == "train":
        keep = tp < TRAIN_TARGET_BEFORE
    else:
        keep = (
            (p >= 201801)
            & (p <= 202012)
            & (tp < VALID_TARGET_BEFORE)
        )

    if not keep.any():
        return None

    frame = frame.loc[keep].copy()
    p, tp = p[keep], tp[keep]

    s = state_codes(frame["state_t"])
    y = state_codes(frame["next_state_t1"])
    risk = binary_codes(frame["first90_risk_set"])
    first = binary_codes(frame["first90_next"])

    month_no = p // 100 * 12 + p % 100
    next_no = tp // 100 * 12 + tp % 100

    good = (
        (p % 100 >= 1)
        & (p % 100 <= 12)
        & (tp % 100 >= 1)
        & (tp % 100 <= 12)
        & (next_no == month_no + 1)
        & (s >= 0)
        & (s <= 4)
        & (y >= 0)
        & (y <= 7)
        & ~((s == 4) & (y == 3))
        & ((risk == 0) | (risk == 1))
        & ~((risk == 1) & (first != (y == 3)))
    )

    if not np.all(good):
        raise ValueError(
            f"{split}: {(~good).sum()} invalid rows"
        )

    return frame, s, y, risk, first, p


def iterate_train():
    for split in ("early_train", "validation"):
        for f in batches(split):
            result = select(f, "train")
            if result is not None:
                yield result


def numeric_array(s):
    a = pd.to_numeric(
        s, errors="coerce"
    ).to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    a[~np.isfinite(a)] = np.nan
    return a


def make_spec():
    print(
        "\n=== PASS 1: TRAINING PREPROCESSING ===",
        flush=True,
    )

    seen = np.zeros(len(NUM), dtype=np.int64)
    sums = np.zeros(len(NUM))
    squares = np.zeros(len(NUM))

    cats = {
        c: {"__MISSING__", "__UNSEEN__"}
        for c in CAT
    }

    total = 0

    for f, _, _, _, _, _ in iterate_train():
        total += len(f)

        for j, c in enumerate(NUM):
            v = numeric_array(f[c])
            ok = np.isfinite(v)

            seen[j] += ok.sum()
            sums[j] += v[ok].sum()
            squares[j] += np.square(v[ok]).sum()

        for c in CAT:
            cats[c].update(
                pd.unique(cat_strings(f[c])).tolist()
            )

    retained = [
        i for i in range(len(NUM))
        if seen[i] > 0
    ]

    dropped = [
        NUM[i] for i in range(len(NUM))
        if seen[i] == 0
    ]

    spec = {
        "rows": int(total),
        "num": {},
        "cat": {},
    }

    for i in retained:
        mu = sums[i] / seen[i]

        var = max(
            squares[i] / seen[i] - mu * mu,
            0.0,
        )

        sd = max(
            float(np.sqrt(var)),
            1e-8,
        )

        spec["num"][NUM[i]] = {
            "mean": float(mu),
            "std": sd,
            "missing_indicator": bool(
                seen[i] < total
            ),
        }

    for c in CAT:
        labels = sorted(cats[c])

        if len(labels) > 250:
            raise ValueError(
                f"Unexpected high cardinality: {c}"
            )

        spec["cat"][c] = labels

    spec["dropped_all_missing"] = dropped

    spec["features"] = (
        list(spec["num"].keys())
        + [
            c + "__missing"
            for c, info in spec["num"].items()
            if info["missing_indicator"]
        ]
        + [
            c + "=" + label
            for c, levels in spec["cat"].items()
            for label in levels
        ]
        + ["__intercept__"]
    )

    print(f"Training rows: {total:,}")
    print(f"Design width: {len(spec['features'])}")
    print(f"All-missing columns dropped: {dropped}")

    return spec


def design(frame, spec):
    n = len(frame)

    cols = []
    miss = []

    for c, info in spec["num"].items():
        a = numeric_array(frame[c])
        missing = ~np.isfinite(a)

        a[missing] = info["mean"]
        a = (a - info["mean"]) / info["std"]

        cols.append(a)

        if info["missing_indicator"]:
            miss.append(
                missing.astype(np.float64)
            )

    numeric = (
        np.column_stack(cols + miss)
        if cols or miss
        else np.empty((n, 0))
    )

    left = sparse.csr_matrix(numeric)

    row_ids = []
    col_ids = []
    offset = 0

    for c, levels in spec["cat"].items():
        strings = cat_strings(frame[c])

        codes = pd.Categorical(
            strings, categories=levels
        ).codes.astype(np.int32)

        codes[codes < 0] = levels.index(
            "__UNSEEN__"
        )

        row_ids.append(
            np.arange(n, dtype=np.int32)
        )
        col_ids.append(codes + offset)

        offset += len(levels)

    cat = sparse.csr_matrix(
        (
            np.ones(
                n * len(CAT), dtype=np.float64
            ),
            (
                np.concatenate(row_ids),
                np.concatenate(col_ids),
            ),
        ),
        shape=(n, offset),
        dtype=np.float64,
    )

    intercept = sparse.csr_matrix(
        np.ones((n, 1))
    )

    x = sparse.hstack(
        (left, cat, intercept),
        format="csr",
    )

    if x.shape[1] != len(spec["features"]):
        raise ValueError(
            "Design matrix width mismatch"
        )

    return x


def probabilities(x, theta, s):
    # S0 is the reference outcome.
    z = np.column_stack([
        np.zeros(x.shape[0]),
        x @ theta,
    ])

    # S4 -> S3 is prohibited.
    z[s == 4, 3] = -1e30

    return softmax(z, axis=1)


def fit(spec):
    print(
        "\n=== FULL-DATA L-BFGS TRAINING ===",
        flush=True,
    )

    n = spec["rows"]
    d = len(spec["features"])

    steps = [0]
    start = time.perf_counter()

    def objective(flat):
        theta = flat.reshape(d, 7)

        loss = 0.0
        grad = np.zeros(
            (d, 7), dtype=np.float64
        )
        rows = 0

        for f, s, y, _, _, _ in iterate_train():
            x = design(f, spec)
            pr = probabilities(x, theta, s)

            idx = np.arange(len(y))

            loss -= np.log(
                np.maximum(
                    pr[idx, y],
                    1e-300,
                )
            ).sum()

            residual = pr[:, 1:].copy()
            positive = y > 0

            residual[
                idx[positive],
                y[positive] - 1,
            ] -= 1.0

            grad += np.asarray(
                x.T @ residual
            )

            rows += len(y)

        if rows != n:
            raise ValueError(
                "Training row count changed"
            )

        loss = (
            loss / n
            + 0.5 * L2
            * np.square(theta[:-1]).sum()
        )

        grad = grad / n
        grad[:-1] += L2 * theta[:-1]

        steps[0] += 1

        if steps[0] == 1 or steps[0] % 5 == 0:
            elapsed = (
                time.perf_counter() - start
            )

            print(
                f"Full pass {steps[0]:3d} | "
                f"Objective={loss:.7f} | "
                f"Elapsed={elapsed:.1f}s",
                flush=True,
            )

        return float(loss), grad.ravel()

    checkpoint_path = (
        OUT / "optimizer_checkpoint.npz"
    )

    def checkpoint(flat):
        np.savez_compressed(
            checkpoint_path,
            theta=flat.reshape(d, 7),
        )

    if checkpoint_path.exists():
        init = np.load(
            checkpoint_path
        )["theta"]

        if init.shape != (d, 7):
            raise ValueError(
                "Incompatible checkpoint shape"
            )

        print(
            "Resuming from optimizer checkpoint",
            flush=True,
        )
    else:
        init = np.zeros(
            (d, 7), dtype=np.float64
        )

    result = minimize(
        objective,
        init.ravel(),
        method="L-BFGS-B",
        jac=True,
        callback=checkpoint,
        options={
            "maxiter": MAX_ITER,
            "ftol": 1e-9,
            "gtol": 1e-5,
            "maxls": 20,
        },
    )

    theta = result.x.reshape(d, 7)

    report = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "full_passes": int(steps[0]),
        "objective": float(result.fun),
        "elapsed_seconds": round(
            time.perf_counter() - start, 1
        ),
        "gradient_max_abs": float(
            np.max(np.abs(result.jac))
        ),
        "train_rows": n,
        "l2": L2,
    }

    np.savez_compressed(
        OUT / "unified_softmax_coefficients.npz",
        theta=theta,
    )

    (OUT / "optimization.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== OPTIMIZATION ===")
    print(
        json.dumps(report, indent=2),
        flush=True,
    )

    return theta, report


def validate(spec, theta):
    print(
        "\n=== VALIDATION 2018-2020 ===",
        flush=True,
    )

    n = 0
    ll_total = 0.0
    bs_total = 0.0

    actual = np.zeros(8)
    predicted = np.zeros(8)

    # n, risk n, first actual/expected,
    # credit actual/expected, payoff actual/expected
    annual = np.zeros((3, 8))

    # count, logloss sum, brier sum
    by_start = np.zeros((5, 3))

    for raw in batches("late_development"):
        result = select(raw, "validate")

        if result is None:
            continue

        f, s, y, risk, first, month = result

        x = design(f, spec)
        pr = probabilities(x, theta, s)

        idx = np.arange(len(y))
        p_true = np.maximum(
            pr[idx, y], 1e-300
        )

        ll = -np.log(p_true)

        bs = (
            1.0
            - 2.0 * p_true
            + np.square(pr).sum(axis=1)
        )

        n += len(y)
        ll_total += ll.sum()
        bs_total += bs.sum()

        actual += np.bincount(
            y, minlength=8
        )
        predicted += pr.sum(axis=0)

        by_start[:, 0] += np.bincount(
            s, minlength=5
        )
        by_start[:, 1] += np.bincount(
            s, weights=ll, minlength=5
        )
        by_start[:, 2] += np.bincount(
            s, weights=bs, minlength=5
        )

        for k in range(3):
            mask = (
                month // 100 == 2018 + k
            )

            if not mask.any():
                continue

            eligible = mask & (risk == 1)

            annual[k] += [
                mask.sum(),
                eligible.sum(),
                first[eligible].sum(),
                pr[eligible, 3].sum(),
                (y[mask] == 6).sum(),
                pr[mask, 6].sum(),
                (y[mask] == 5).sum(),
                pr[mask, 5].sum(),
            ]

    if n != 10_344_541:
        raise ValueError(
            f"Validation rows {n:,} "
            "disagree with Task 29"
        )

    def oe(a, e):
        return (
            float(a / e)
            if e > 0
            else None
        )

    overall = {
        "validation_rows": n,
        "multiclass_logloss": float(
            ll_total / n
        ),
        "multiclass_brier": float(
            bs_total / n
        ),
        "first90_actual": int(
            annual[:, 2].sum()
        ),
        "first90_expected": float(
            annual[:, 3].sum()
        ),
        "first90_OE": oe(
            annual[:, 2].sum(),
            annual[:, 3].sum(),
        ),
        "credit_exit_OE": oe(
            actual[6], predicted[6]
        ),
        "payoff_OE": oe(
            actual[5], predicted[5]
        ),
    }

    table = []

    for k in range(3):
        r = annual[k]

        table.append({
            "year": 2018 + k,
            "rows": int(r[0]),
            "first90_risk_rows": int(r[1]),
            "first90_actual": int(r[2]),
            "first90_expected": float(r[3]),
            "first90_OE": oe(r[2], r[3]),
            "credit_exit_OE": oe(r[4], r[5]),
            "payoff_OE": oe(r[6], r[7]),
        })

    pd.DataFrame(table).to_csv(
        OUT / "validation_by_year.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "next_state": f"S{k}",
            "actual": int(actual[k]),
            "expected": float(predicted[k]),
            "OE": oe(
                actual[k], predicted[k]
            ),
        }
        for k in range(8)
    ]).to_csv(
        OUT / "validation_by_next_state.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "starting_state": f"S{k}",
            "rows": int(by_start[k, 0]),
            "logloss": (
                float(
                    by_start[k, 1]
                    / by_start[k, 0]
                )
                if by_start[k, 0]
                else None
            ),
            "brier": (
                float(
                    by_start[k, 2]
                    / by_start[k, 0]
                )
                if by_start[k, 0]
                else None
            ),
        }
        for k in range(5)
    ]).to_csv(
        OUT / "validation_by_start_state.csv",
        index=False,
    )

    (OUT / "validation_summary.json").write_text(
        json.dumps(overall, indent=2),
        encoding="utf-8",
    )

    print("\n=== OVERALL ===")
    print(
        pd.DataFrame([overall])
        .round(5)
        .to_string(index=False)
    )

    print("\n=== BY YEAR ===")
    print(
        pd.DataFrame(table)[
            [
                "year",
                "first90_OE",
                "credit_exit_OE",
                "payoff_OE",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    if BASELINE.exists():
        base = pd.read_csv(BASELINE)

        print("\n=== TASK 29 BASELINE ===")
        print(
            base[
                [
                    "model",
                    "multiclass_logloss",
                    "multiclass_brier",
                    "first90_OE",
                    "credit_exit_OE",
                    "payoff_OE",
                ]
            ]
            .round(5)
            .to_string(index=False)
        )

    return overall


def main():
    OUT.mkdir(
        parents=True, exist_ok=True
    )

    t = time.perf_counter()

    print(
        "Task 30 | Full-data multinomial logistic",
        flush=True,
    )

    print(
        "Train: 2006-2017 | "
        "Validate: 2018-2020 | "
        "Test: untouched",
        flush=True,
    )

    spec = make_spec()

    spec_path = OUT / "feature_spec.json"

    if (
        spec_path.exists()
        and (
            OUT / "optimizer_checkpoint.npz"
        ).exists()
    ):
        old_spec = json.loads(
            spec_path.read_text(
                encoding="utf-8"
            )
        )

        if old_spec != spec:
            raise ValueError(
                "Checkpoint feature specification "
                "does not match current data"
            )

    spec_path.write_text(
        json.dumps(spec, indent=2),
        encoding="utf-8",
    )

    theta, optimization = fit(spec)

    if not optimization["success"]:
        print(
            "WARNING: Optimization did not meet "
            "convergence criteria. "
            "Validation is diagnostic only.",
            flush=True,
        )

    validate(spec, theta)

    print(
        f"\nCompleted in "
        f"{(time.perf_counter()-t)/60:.1f} min."
    )

    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()
