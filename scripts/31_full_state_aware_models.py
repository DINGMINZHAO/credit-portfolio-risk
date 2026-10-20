
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize


ROOT = Path(
    os.getenv(
        "CREDIT_PORT_ROOT",
        str(Path.home() / "Desktop/credit-portfolio-data"),
    )
)

SOURCE = ROOT / "scripts/30_full_multinomial_logistic.py"
PREVIOUS = ROOT / "outputs/modeling/task30_streaming_logistic"
OUT = ROOT / "outputs/modeling/task31_state_aware"

MACRO = (
    "hpi_yoy_pct",
    "unemployment_pct",
    "unemployment_change_3m_pp",
    "mortgage30_pct",
    "refinance_incentive_pp",
)

AGE_KNOTS = (12, 36, 60, 120)

COMMON_INTERACTIONS = (
    "orig_credit_score",
    "orig_ltv",
    "loan_age_reported_t",
    "log_upb_t",
    "months_90plus_observed_through_t",
)

M2_ONLY_INTERACTIONS = (
    "refinance_incentive_pp",
    "hpi_yoy_pct",
    "unemployment_pct",
)

L2 = 1e-6


def load_source():
    spec = importlib.util.spec_from_file_location(
        "task30_for_task31", SOURCE
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_previous():
    old_spec = json.loads(
        (PREVIOUS / "feature_spec.json").read_text(
            encoding="utf-8"
        )
    )

    old_theta = np.load(
        PREVIOUS / "unified_softmax_coefficients.npz"
    )["theta"]

    if old_theta.shape != (
        len(old_spec["features"]), 7
    ):
        raise ValueError(
            "Task 30 specification and coefficients disagree"
        )

    return old_spec, old_theta


def spec_for(mode, old):
    original = old["features"]

    reference = {}
    drop = set()

    for cat, levels in old["cat"].items():
        eligible = [
            v for v in levels
            if v not in (
                "__MISSING__",
                "__UNSEEN__",
            )
        ]

        if not eligible:
            eligible = list(levels)

        preferred = ["S0", "0", "False", "false"]

        chosen = next(
            (v for v in preferred if v in eligible),
            eligible[0],
        )

        reference[cat] = f"{cat}={chosen}"
        drop.add(reference[cat])

    if mode == "m3":
        for name in MACRO:
            drop.add(name)
            drop.add(name + "__missing")

    base_names = [
        n for n in original
        if n not in drop
    ]

    if "__intercept__" not in base_names:
        raise ValueError("Missing intercept")

    interact = list(COMMON_INTERACTIONS)

    if mode == "m2":
        interact += list(M2_ONLY_INTERACTIONS)

    hinges = [
        f"age_hinge_{k}m"
        for k in AGE_KNOTS
    ]

    signals = (
        ["ever90_is_1"]
        + list(interact)
        + hinges
    )

    extra = hinges + [
        f"state_S{k}__x__{name}"
        for k in range(1, 5)
        for name in signals
    ]

    all_names = base_names + extra

    if len(all_names) != len(set(all_names)):
        raise ValueError("Duplicate feature names")

    return {
        "model": mode,
        "base_names": base_names,
        "base_indices": [
            original.index(n)
            for n in base_names
        ],
        "reference": reference,
        "interact": interact,
        "hinges": hinges,
        "signals": signals,
        "features": all_names,
        "penalty": L2,
        "train_target_before": 201712,
        "validation_target_before": 202012,
        "n_train": old["rows"],
    }


def start_from_previous(spec, old_spec, old_theta):
    pos = {
        name: i
        for i, name in enumerate(
            old_spec["features"]
        )
    }

    shift = np.zeros(7)

    for ref in spec["reference"].values():
        shift += old_theta[pos[ref], :]

    result = np.zeros((
        len(spec["features"]), 7
    ))

    for j, name in enumerate(
        spec["base_names"]
    ):
        if name == "__intercept__":
            result[j] = (
                old_theta[pos[name], :]
                + shift
            )
            continue

        parent = next(
            (
                c for c in spec["reference"]
                if name.startswith(c + "=")
            ),
            None,
        )

        if parent:
            result[j] = (
                old_theta[pos[name], :]
                - old_theta[
                    pos[spec["reference"][parent]], :
                ]
            )
        else:
            result[j] = old_theta[pos[name], :]

    return result


def design(frame, state, spec, old_spec, module):
    base_all = module.design(
        frame, old_spec
    )

    base = base_all[
        :, spec["base_indices"]
    ]

    n = len(frame)

    age = module.numeric_array(
        frame["loan_age_reported_t"]
    )

    age[~np.isfinite(age)] = old_spec[
        "num"
    ]["loan_age_reported_t"]["mean"]

    age = np.clip(age, 0, 360)

    hinge = np.column_stack([
        np.maximum(age - knot, 0) / 120.0
        for knot in AGE_KNOTS
    ])

    ever_text = module.cat_strings(
        frame["ever_90plus_through_t"]
    ).str.lower()

    ever = ever_text.isin(
        ("1", "true", "yes", "y", "t")
    ).to_numpy(dtype=np.float64)

    original_pos = {
        name: i
        for i, name in enumerate(
            old_spec["features"]
        )
    }

    signals = [ever]

    for name in spec["interact"]:
        signals.append(
            np.asarray(
                base_all[
                    :, original_pos[name]
                ].toarray()
            ).ravel()
        )

    signals.extend([
        hinge[:, j]
        for j in range(len(AGE_KNOTS))
    ])

    signal_matrix = np.column_stack(
        signals
    )

    blocks = [hinge]

    for k in range(1, 5):
        blocks.append(
            signal_matrix
            * (state == k)[:, None]
        )

    extras = sparse.csr_matrix(
        np.column_stack(blocks)
    )

    x = sparse.hstack(
        (base, extras),
        format="csr",
    )

    if x.shape != (
        n, len(spec["features"])
    ):
        raise ValueError(
            "Design dimensions mismatch"
        )

    return x


def objective_batch(
    x, state, y, theta, module
):
    pr = module.probabilities(
        x, theta, state
    )

    row = np.arange(len(y))

    logloss_sum = -np.log(
        np.maximum(
            pr[row, y], 1e-300
        )
    ).sum()

    residual = pr[:, 1:].copy()

    positive = y > 0

    residual[
        row[positive],
        y[positive] - 1,
    ] -= 1

    gradient = np.asarray(
        x.T @ residual
    )

    return logloss_sum, gradient


def numerical_gradient_check(
    module, spec, old_spec
):
    frame, state, y, _, _, _ = next(
        module.iterate_train()
    )

    n = min(256, len(y))

    state = state[:n].copy()
    y = y[:n].copy()

    frame = frame.iloc[:n]

    state[0] = 4
    y[0] = 0

    x = design(
        frame, state, spec,
        old_spec, module
    )

    rng = np.random.default_rng(3102)

    theta = rng.normal(
        0, 0.025,
        size=(x.shape[1], 7),
    )

    penalized = np.array(
        [
            name != "__intercept__"
            for name in spec["features"]
        ],
        dtype=float,
    )

    n = len(y)

    def fun(flat):
        t = flat.reshape(
            x.shape[1], 7
        )

        l, g = objective_batch(
            x, state, y, t, module
        )

        l = (
            l / n
            + 0.5 * L2
            * np.square(
                t * penalized[:, None]
            ).sum()
        )

        g = (
            g / n
            + L2 * t * penalized[:, None]
        )

        return l, g.ravel()

    flat = theta.ravel()
    analytical = fun(flat)[1]

    chosen = rng.choice(
        len(flat),
        min(20, len(flat)),
        replace=False,
    )

    eps = 1e-5
    discrepancies = []

    for k in chosen:
        plus = flat.copy()
        minus = flat.copy()

        plus[k] += eps
        minus[k] -= eps

        approximate = (
            fun(plus)[0]
            - fun(minus)[0]
        ) / (2 * eps)

        discrepancies.append(
            abs(
                approximate
                - analytical[k]
            )
        )

    diff = max(discrepancies)

    forbidden = module.probabilities(
        x, theta, state
    )[0, 3]

    if diff > 1e-5 or forbidden != 0:
        raise ValueError(
            "Gradient/constraint check failed: "
            f"{diff}, {forbidden}"
        )

    print(
        "Gradient check OK: "
        f"max abs error={diff:.2e}; "
        f"S4->S3={forbidden:.0f}",
        flush=True,
    )


def fit_one(
    mode, spec, old_spec, old_theta,
    module, maxiter, force=False,
):
    folder = OUT / mode

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        folder / "coefficients.npz"
    )

    spec_path = (
        folder / "model_spec.json"
    )

    checkpoint_path = (
        folder / "checkpoint.npz"
    )

    content = json.dumps(
        spec,
        indent=2,
        ensure_ascii=False,
    )

    if (
        spec_path.exists()
        and spec_path.read_text(
            encoding="utf-8"
        ) != content
    ):
        raise ValueError(
            f"{mode}: model specification differs; "
            "refusing unsafe resume"
        )

    spec_path.write_text(
        content,
        encoding="utf-8",
    )

    if model_path.exists() and not force:
        report_path = (
            folder / "optimization.json"
        )

        previous_report = (
            json.loads(
                report_path.read_text(
                    encoding="utf-8"
                )
            )
            if report_path.exists()
            else {}
        )

        if previous_report.get(
            "success", False
        ):
            print(
                f"{mode}: converged model exists; "
                "skipping retraining",
                flush=True,
            )
            return

        print(
            f"{mode}: previous run not converged; "
            "continuing from checkpoint",
            flush=True,
        )

    numerical_gradient_check(
        module, spec, old_spec
    )

    d = len(spec["features"])

    mask = np.array(
        [
            name != "__intercept__"
            for name in spec["features"]
        ],
        dtype=np.float64,
    )[:, None]

    if (
        checkpoint_path.exists()
        and not force
    ):
        init = np.load(
            checkpoint_path
        )["theta"]

        print(
            f"{mode}: warm restart from checkpoint "
            "(L-BFGS history resets)",
            flush=True,
        )
    else:
        init = start_from_previous(
            spec, old_spec, old_theta
        )

        print(
            f"{mode}: initialized from Task 30",
            flush=True,
        )

    if init.shape != (d, 7):
        raise ValueError(
            "Initialization shape mismatch"
        )

    start = time.perf_counter()
    passes = [0]

    def objective(flat):
        theta = flat.reshape(d, 7)

        total_loss = 0.0

        total_gradient = np.zeros(
            (d, 7), dtype=np.float64
        )

        total_n = 0

        for frame, state, y, _, _, _ in (
            module.iterate_train()
        ):
            x = design(
                frame, state, spec,
                old_spec, module
            )

            loss, gradient = objective_batch(
                x, state, y, theta, module
            )

            total_loss += loss
            total_gradient += gradient
            total_n += len(y)

        if total_n != spec["n_train"]:
            raise ValueError(
                f"Train count changed: {total_n}"
            )

        penalty = (
            0.5 * L2
            * np.square(theta * mask).sum()
        )

        loss = (
            total_loss / total_n
            + penalty
        )

        gradient = (
            total_gradient / total_n
            + L2 * theta * mask
        )

        passes[0] += 1

        if (
            passes[0] == 1
            or passes[0] % 10 == 0
        ):
            print(
                f"{mode} pass {passes[0]:3d} | "
                f"obj={loss:.7f} | "
                f"elapsed="
                f"{(time.perf_counter()-start)/60:.1f} min",
                flush=True,
            )

        return float(loss), gradient.ravel()

    def checkpoint(flat):
        temp = (
            folder / "checkpoint.tmp.npz"
        )

        np.savez_compressed(
            temp,
            theta=flat.reshape(d, 7),
        )

        os.replace(
            temp, checkpoint_path
        )

    result = minimize(
        objective,
        init.ravel(),
        method="L-BFGS-B",
        jac=True,
        callback=checkpoint,
        options={
            "maxiter": maxiter,
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 25,
        },
    )

    theta = result.x.reshape(d, 7)

    checkpoint(theta.ravel())

    np.savez_compressed(
        model_path,
        theta=theta,
    )

    report = {
        "model": mode,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations_this_run": int(result.nit),
        "full_passes_this_run": passes[0],
        "objective": float(result.fun),
        "max_abs_gradient": float(
            np.max(np.abs(result.jac))
        ),
        "elapsed_minutes_this_run": round(
            (time.perf_counter()-start)/60,
            1,
        ),
        "n_train": spec["n_train"],
        "l2": L2,
        "warm_start_note": (
            "Checkpoint restart does not retain "
            "L-BFGS curvature history"
        ),
    }

    (folder / "optimization.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n{mode} optimization:\n"
        + json.dumps(report, indent=2),
        flush=True,
    )

    if not result.success:
        print(
            f"WARNING: {mode} did not converge; "
            "validation remains diagnostic",
            flush=True,
        )


def evaluate_one(
    mode, spec, old_spec, module
):
    theta = np.load(
        OUT / mode / "coefficients.npz"
    )["theta"]

    if theta.shape != (
        len(spec["features"]), 7
    ):
        raise ValueError(
            f"{mode}: coefficient dimension mismatch"
        )

    def empty_bucket():
        return {
            "rows": 0,
            "ll": 0.0,
            "bs": 0.0,
            "actual": np.zeros(8),
            "expected": np.zeros(8),
            "risk_rows": 0,
            "first_actual": 0.0,
            "first_expected": 0.0,
        }

    buckets = defaultdict(empty_bucket)

    def push(
        label, y, prob, risk,
        first, ll, bs,
    ):
        if not len(y):
            return

        b = buckets[label]

        b["rows"] += len(y)
        b["ll"] += float(ll.sum())
        b["bs"] += float(bs.sum())

        b["actual"] += np.bincount(
            y, minlength=8
        )

        b["expected"] += prob.sum(
            axis=0
        )

        eligible = risk == 1

        b["risk_rows"] += int(
            eligible.sum()
        )

        b["first_actual"] += float(
            first[eligible].sum()
        )

        b["first_expected"] += float(
            prob[eligible, 3].sum()
        )

    for raw in module.batches(
        "late_development"
    ):
        item = module.select(
            raw, "validate"
        )

        if item is None:
            continue

        frame, state, y, risk, first, month = item

        x = design(
            frame, state, spec,
            old_spec, module
        )

        prob = module.probabilities(
            x, theta, state
        )

        row = np.arange(len(y))

        pt = prob[row, y]

        ll = -np.log(
            np.maximum(pt, 1e-300)
        )

        bs = (
            1.0
            - 2.0 * pt
            + np.square(prob).sum(axis=1)
        )

        push(
            "all", y, prob,
            risk, first, ll, bs
        )

        years = month // 100

        for year in np.unique(years):
            which = years == year

            push(
                f"year_{year}",
                y[which],
                prob[which],
                risk[which],
                first[which],
                ll[which],
                bs[which],
            )

        for s in range(5):
            which = state == s

            push(
                f"start_S{s}",
                y[which],
                prob[which],
                risk[which],
                first[which],
                ll[which],
                bs[which],
            )

    if buckets["all"]["rows"] != 10_344_541:
        raise ValueError(
            f"{mode}: validation count mismatch"
        )

    def oe(a, e):
        return (
            float(a / e)
            if e > 0
            else np.nan
        )

    records = []

    for group, b in sorted(buckets.items()):
        r = {
            "model": mode,
            "group": group,
            "rows": b["rows"],
            "logloss": (
                b["ll"] / b["rows"]
            ),
            "brier": (
                b["bs"] / b["rows"]
            ),
            "first90_actual": int(
                b["first_actual"]
            ),
            "first90_expected": b[
                "first_expected"
            ],
            "first90_OE": oe(
                b["first_actual"],
                b["first_expected"],
            ),
        }

        for j in range(8):
            r[f"S{j}_actual"] = int(
                b["actual"][j]
            )

            r[f"S{j}_expected"] = float(
                b["expected"][j]
            )

            r[f"S{j}_OE"] = oe(
                b["actual"][j],
                b["expected"][j],
            )

        records.append(r)

    result = pd.DataFrame(records)

    result.to_csv(
        OUT / mode / "validation_breakdown.csv",
        index=False,
    )

    cols = [
        "model",
        "group",
        "rows",
        "logloss",
        "brier",
        "first90_OE",
        "S5_OE",
        "S6_OE",
    ]

    print(
        f"\n=== {mode.upper()} VALIDATION ==="
    )

    selected = result[
        result["group"].str.startswith(
            ("all", "year_")
        )
    ]

    print(
        selected[cols]
        .round(5)
        .to_string(index=False),
        flush=True,
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=("m2", "m3", "both"),
        default="both",
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart from Task 30 parameters",
    )

    args = parser.parse_args()

    if args.maxiter < 1:
        parser.error(
            "--maxiter must be >= 1"
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    module = load_source()
    old_spec, old_theta = load_previous()

    if (
        module.TRAIN_TARGET_BEFORE != 201712
        or module.VALID_TARGET_BEFORE != 202012
    ):
        raise ValueError(
            "Task 30 date cutoffs have changed"
        )

    modes = (
        ("m2", "m3")
        if args.model == "both"
        else (args.model,)
    )

    for mode in modes:
        spec = spec_for(
            mode, old_spec
        )

        print(
            f"\n### {mode.upper()} FULL-DATA TRAINING "
            f"({spec['n_train']:,} rows, "
            f"{len(spec['features'])} features) ###",
            flush=True,
        )

        fit_one(
            mode,
            spec,
            old_spec,
            old_theta,
            module,
            args.maxiter,
            force=args.force,
        )

        evaluate_one(
            mode,
            spec,
            old_spec,
            module,
        )


if __name__ == "__main__":
    main()
