
from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    os.getenv(
        "CREDIT_PORT_ROOT",
        str(Path.home() / "Desktop/credit-portfolio-data"),
    )
)

SOURCE = ROOT / "scripts/30_full_multinomial_logistic.py"
OUT = ROOT / "outputs/modeling/task30_streaming_logistic"
DIAG = OUT / "diagnostics"


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "task30_source", path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def safe_ratio(actual, expected):
    return (
        float(actual / expected)
        if expected > 0
        else np.nan
    )


def fresh():
    return {
        "rows": 0,
        "ll_sum": 0.0,
        "bs_sum": 0.0,
        "actual": np.zeros(8),
        "expected": np.zeros(8),
        "first90_risk_rows": 0,
        "first90_actual": 0.0,
        "first90_expected": 0.0,
        "first90_ll_sum": 0.0,
        "first90_bs_sum": 0.0,
        "floored_true": 0,
    }


def update(bucket, y, p, risk, first, ll, bs, floored):
    if not len(y):
        return

    n = len(y)

    bucket["rows"] += n
    bucket["ll_sum"] += float(ll.sum())
    bucket["bs_sum"] += float(bs.sum())

    bucket["actual"] += np.bincount(
        y, minlength=8
    )

    bucket["expected"] += p.sum(axis=0)

    bucket["floored_true"] += int(floored.sum())

    eligible = risk == 1

    if eligible.any():
        event = first[eligible].astype(np.float64)
        probability = p[eligible, 3]

        q = np.clip(
            probability, 1e-300, 1 - 1e-15
        )

        bucket["first90_risk_rows"] += int(
            eligible.sum()
        )

        bucket["first90_actual"] += float(
            event.sum()
        )

        bucket["first90_expected"] += float(
            probability.sum()
        )

        bucket["first90_ll_sum"] += float(
            (
                -(
                    event * np.log(q)
                    + (1 - event) * np.log1p(-q)
                )
            ).sum()
        )

        bucket["first90_bs_sum"] += float(
            np.square(probability - event).sum()
        )


def summary(bucket):
    n = bucket["rows"]
    r = bucket["first90_risk_rows"]

    a = bucket["actual"]
    e = bucket["expected"]

    result = {
        "rows": n,
        "logloss": (
            bucket["ll_sum"] / n if n else np.nan
        ),
        "brier": (
            bucket["bs_sum"] / n if n else np.nan
        ),
    }

    for k in range(8):
        result[f"S{k}_OE"] = safe_ratio(
            a[k], e[k]
        )

    for k in (5, 6):
        result[f"S{k}_actual"] = int(a[k])
        result[f"S{k}_expected"] = float(e[k])

    result.update({
        "first90_risk_rows": r,
        "first90_actual": int(
            bucket["first90_actual"]
        ),
        "first90_expected": bucket["first90_expected"],
        "first90_OE": safe_ratio(
            bucket["first90_actual"],
            bucket["first90_expected"],
        ),
        "first90_binary_logloss": (
            bucket["first90_ll_sum"] / r
            if r else np.nan
        ),
        "first90_binary_brier": (
            bucket["first90_bs_sum"] / r
            if r else np.nan
        ),
        "floored_true": bucket["floored_true"],
    })

    return result


def gradient_check(mod, model_spec):
    frame, state, label, _, _, _ = next(
        mod.iterate_train()
    )

    n = min(len(label), 512)

    x = mod.design(
        frame.iloc[:n], model_spec
    )

    s = state[:n].copy()
    y = label[:n].copy()

    # Synthetic admissible S4 case for constraint testing.
    s[0] = 4
    y[0] = 0

    d = x.shape[1]
    rng = np.random.default_rng(20261018)

    theta = rng.normal(
        0, 0.03, size=(d, 7)
    )

    row = np.arange(n)

    def loss_and_gradient(v):
        t = v.reshape(d, 7)

        p = mod.probabilities(x, t, s)
        probability = p[row, y]

        value = -np.log(
            np.maximum(probability, 1e-300)
        ).mean()

        value += (
            0.5 * mod.L2
            * np.square(t[:-1]).sum()
        )

        residual = p[:, 1:].copy()
        positive = y > 0

        residual[
            row[positive],
            y[positive] - 1,
        ] -= 1

        grad = np.asarray(
            x.T @ residual
        ) / n

        grad[:-1] += mod.L2 * t[:-1]

        return value, grad.ravel()

    flat = theta.ravel()

    _, analytical = loss_and_gradient(flat)

    selected = rng.choice(
        flat.size,
        size=min(24, flat.size),
        replace=False,
    )

    selected = np.unique(
        np.r_[
            selected,
            (d - 1) * 7,
            (d - 1) * 7 + 2,
        ]
    )

    eps = 1e-5
    max_diff = 0.0
    max_scaled_diff = 0.0

    for idx in selected:
        plus = flat.copy()
        minus = flat.copy()

        plus[idx] += eps
        minus[idx] -= eps

        numeric = (
            loss_and_gradient(plus)[0]
            - loss_and_gradient(minus)[0]
        ) / (2 * eps)

        difference = abs(
            numeric - analytical[idx]
        )

        max_diff = max(
            max_diff, difference
        )

        max_scaled_diff = max(
            max_scaled_diff,
            difference / max(
                1.0,
                abs(numeric),
                abs(analytical[idx]),
            ),
        )

    p = mod.probabilities(x, theta, s)

    result = {
        "checked_coordinates": int(len(selected)),
        "max_absolute_gradient_error": float(max_diff),
        "max_scaled_gradient_error": float(
            max_scaled_diff
        ),
        "max_probability_sum_error": float(
            np.abs(p.sum(axis=1) - 1).max()
        ),
        "S4_to_S3_probability": float(p[0, 3]),
        "pass": bool(
            max_scaled_diff < 1e-5
            and p[0, 3] == 0
        ),
    }

    if not result["pass"]:
        raise RuntimeError(
            f"Gradient check failed: {result}"
        )

    return result


def loan_age_groups(frame):
    age = pd.to_numeric(
        frame["loan_age_reported_t"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    valid = np.isfinite(age)

    return {
        "00-11": valid & (age >= 0) & (age < 12),
        "12-35": valid & (age >= 12) & (age < 36),
        "36-59": valid & (age >= 36) & (age < 60),
        "60-119": valid & (age >= 60) & (age < 120),
        "120+": valid & (age >= 120),
        "missing_or_invalid": ~valid | (age < 0),
    }


def scan(split, source, spec, theta, buckets):
    if split == "train":
        iterator = source.iterate_train()

    else:
        def valid_items():
            for raw in source.batches(
                "late_development"
            ):
                result = source.select(
                    raw, "validate"
                )

                if result is not None:
                    yield result

        iterator = valid_items()

    n_processed = 0

    for frame, state, y, risk, first, month in iterator:
        x = source.design(frame, spec)

        p = source.probabilities(
            x, theta, state
        )

        n = len(y)
        indices = np.arange(n)

        ptrue = p[indices, y]
        floored = ptrue <= 1e-300

        ll = -np.log(
            np.maximum(ptrue, 1e-300)
        )

        bs = (
            1
            - 2 * ptrue
            + np.square(p).sum(axis=1)
        )

        if (
            not np.isfinite(ll).all()
            or not np.isfinite(bs).all()
        ):
            raise RuntimeError(
                f"Non-finite losses in {split}"
            )

        if not np.allclose(
            p.sum(axis=1), 1.0, atol=1e-10
        ):
            raise RuntimeError(
                "Probabilities do not sum to 1"
            )

        if np.any(p[state == 4, 3] != 0):
            raise RuntimeError(
                "S4 -> S3 probability not zero"
            )

        if np.any(
            (risk == 1)
            & (first != (y == 3))
        ):
            raise RuntimeError(
                "First90 labels inconsistent"
            )

        def add(group_type, group, mask):
            if mask.any():
                update(
                    buckets[
                        (split, group_type, str(group))
                    ],
                    y[mask],
                    p[mask],
                    risk[mask],
                    first[mask],
                    ll[mask],
                    bs[mask],
                    floored[mask],
                )

        add(
            "overall", "ALL",
            np.ones(n, dtype=bool)
        )

        # Match Task 30: predictor calendar year.
        years = month // 100

        for year in np.unique(years):
            add(
                "year", int(year),
                years == year
            )

        for start in range(5):
            add(
                "start_state",
                f"S{start}",
                state == start,
            )

        for name, mask in loan_age_groups(frame).items():
            add(
                "loan_age", name, mask
            )

        previous = n_processed
        n_processed += n

        if (
            n_processed // 5_000_000
            > previous // 5_000_000
        ):
            print(
                f"{split}: {n_processed:,} rows",
                flush=True,
            )

    return n_processed


def save_results(buckets):
    records = []
    states = []

    for (split, group_type, group), acc in sorted(
        buckets.items()
    ):
        metrics = summary(acc)

        records.append({
            "split": split,
            "group_type": group_type,
            "group": group,
            **metrics,
        })

        for k in range(8):
            states.append({
                "split": split,
                "group_type": group_type,
                "group": group,
                "next_state": f"S{k}",
                "actual": int(acc["actual"][k]),
                "expected": float(
                    acc["expected"][k]
                ),
                "OE": safe_ratio(
                    acc["actual"][k],
                    acc["expected"][k],
                ),
            })

    detail = pd.DataFrame(records)

    detail.to_csv(
        DIAG / "group_metrics.csv",
        index=False,
    )

    pd.DataFrame(states).to_csv(
        DIAG / "state_calibration.csv",
        index=False,
    )

    outputs = {
        "overall": "overall.csv",
        "year": "by_year.csv",
        "start_state": "by_start_state.csv",
        "loan_age": "by_loan_age.csv",
    }

    for kind, filename in outputs.items():
        detail[
            detail.group_type == kind
        ].to_csv(
            DIAG / filename,
            index=False,
        )

    return detail


def main():
    start = time.perf_counter()

    DIAG.mkdir(
        parents=True, exist_ok=True
    )

    mod = load_module(SOURCE)

    spec = json.loads(
        (OUT / "feature_spec.json").read_text(
            encoding="utf-8"
        )
    )

    theta = np.load(
        OUT / "unified_softmax_coefficients.npz"
    )["theta"]

    optimization = json.loads(
        (OUT / "optimization.json").read_text(
            encoding="utf-8"
        )
    )

    if theta.shape != (
        len(spec["features"]), 7
    ):
        raise ValueError(
            "Feature / coefficient shape mismatch"
        )

    if spec["rows"] != optimization["train_rows"]:
        raise ValueError(
            "Training row counts disagree"
        )

    if not np.isfinite(theta).all():
        raise ValueError(
            "Non-finite coefficients"
        )

    check = gradient_check(mod, spec)

    (DIAG / "gradient_check.json").write_text(
        json.dumps(check, indent=2)
    )

    print(
        "\n=== GRADIENT CHECK ==="
    )
    print(json.dumps(check, indent=2))

    buckets = defaultdict(fresh)

    for split in ("train", "validation"):
        n = scan(
            split, mod, spec, theta, buckets
        )

        if split == "train" and n != spec["rows"]:
            raise ValueError(
                f"Train row mismatch: {n}"
            )

        if (
            split == "validation"
            and n != 10_344_541
        ):
            raise ValueError(
                f"Validation row mismatch: {n}"
            )

    detail = save_results(buckets)

    train = buckets[
        ("train", "overall", "ALL")
    ]

    valid = buckets[
        ("validation", "overall", "ALL")
    ]

    objective_reconstructed = (
        train["ll_sum"] / train["rows"]
        + 0.5 * mod.L2
        * np.square(theta[:-1]).sum()
    )

    objective_difference = abs(
        objective_reconstructed
        - optimization["objective"]
    )

    if objective_difference > 1e-6:
        raise RuntimeError(
            "Saved training objective mismatch: "
            f"{objective_difference}"
        )

    previous = json.loads(
        (OUT / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )

    current = summary(valid)

    validation_checks = {
        "multiclass_logloss": current["logloss"],
        "multiclass_brier": current["brier"],
        "first90_OE": current["first90_OE"],
        "credit_exit_OE": current["S6_OE"],
        "payoff_OE": current["S5_OE"],
    }

    for key, calculated in validation_checks.items():
        if abs(calculated - previous[key]) > 1e-5:
            raise RuntimeError(
                f"Validation mismatch: {key}: "
                f"{calculated} vs {previous[key]}"
            )

    checks = {
        "gradient": check,
        "train_objective_reconstructed": float(
            objective_reconstructed
        ),
        "train_objective_saved": float(
            optimization["objective"]
        ),
        "train_objective_absolute_gap": float(
            objective_difference
        ),
        "validation_summary_reproduced": True,
        "run_seconds": round(
            time.perf_counter() - start, 1
        ),
    }

    (DIAG / "checks.json").write_text(
        json.dumps(checks, indent=2)
    )

    print("\n=== TRAIN / VALIDATION ===")

    cols = [
        "split", "rows",
        "logloss", "brier",
        "first90_OE", "S5_OE",
        "S6_OE", "floored_true",
    ]

    print(
        detail.loc[
            detail.group_type == "overall",
            cols,
        ].round(5).to_string(index=False)
    )

    print("\n=== VALIDATION BY YEAR ===")

    frame = detail[
        (detail.group_type == "year")
        & (detail.split == "validation")
    ]

    print(
        frame[
            [
                "group", "rows",
                "logloss", "brier",
                "first90_OE", "S5_OE", "S6_OE",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n=== VALIDATION BY START STATE ===")

    frame = detail[
        (detail.group_type == "start_state")
        & (detail.split == "validation")
    ]

    print(
        frame[
            [
                "group", "rows",
                "logloss", "brier",
                "first90_OE", "S5_OE", "S6_OE",
            ]
        ].round(5).to_string(index=False)
    )

    print(
        "\nObjective reproduction gap:",
        f"{objective_difference:.3g}",
    )

    print(
        "Results saved to:", DIAG
    )


if __name__ == "__main__":
    main()
