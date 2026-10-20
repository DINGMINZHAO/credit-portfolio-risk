
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.home() / "Desktop/credit-portfolio-data"
DATA = ROOT / "data/processed/model_ready/28_monthly_transitions"
OUT = ROOT / "outputs/modeling/task29_benchmark"

COLUMNS = [
    "period_yyyymm",
    "target_period_yyyymm",
    "state_t",
    "next_state_t1",
    "ever_90plus_through_t",
    "first90_risk_set",
    "first90_next",
]

BATCH_SIZE = 120_000
SHRINKAGE = 200.0

# Conservative buffer for monthly reporting delays.
TRAIN_TARGET_BEFORE = 201712
VALID_TARGET_BEFORE = 202012

# S0-S4 are active states; S5-S7 are absorbing exits.
ALLOWED = np.ones((5, 8), dtype=bool)
ALLOWED[4, 3] = False


def batches(split):
    import pyarrow.parquet as pq

    files = sorted((DATA / split).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files in {DATA / split}"
        )

    print(
        f"  {split}: {len(files)} files",
        flush=True,
    )

    for file in files:
        pf = pq.ParquetFile(file)

        missing = set(COLUMNS) - set(
            pf.schema_arrow.names
        )

        if missing:
            raise ValueError(
                f"{file.name}: missing {sorted(missing)}"
            )

        for batch in pf.iter_batches(
            columns=COLUMNS,
            batch_size=BATCH_SIZE,
        ):
            yield batch.to_pandas()


def number(series, missing=-1):
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(missing)
        .to_numpy()
    )


def state_codes(series):
    text = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^S", "", regex=True)
    )
    return number(text).astype(np.int16)


def binary_codes(series):
    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "true": "1",
            "false": "0",
        })
    )
    return number(text).astype(np.int8)


def parse(frame):
    state = state_codes(frame["state_t"])
    next_state = state_codes(frame["next_state_t1"])

    history = binary_codes(
        frame["ever_90plus_through_t"]
    )
    risk = binary_codes(
        frame["first90_risk_set"]
    )
    first = binary_codes(
        frame["first90_next"]
    )

    period = number(
        frame["period_yyyymm"]
    ).astype(np.int32)

    target_period = number(
        frame["target_period_yyyymm"]
    ).astype(np.int32)

    valid = (
        (state >= 0)
        & (state <= 4)
        & (next_state >= 0)
        & (next_state <= 7)
        & (period % 100 >= 1)
        & (period % 100 <= 12)
        & (target_period % 100 >= 1)
        & (target_period % 100 <= 12)
        & (
            (
                target_period // 100 * 12
                + target_period % 100
            )
            == (
                period // 100 * 12
                + period % 100 + 1
            )
        )
        & np.isin(risk, [0, 1])
    )

    if not valid.all():
        raise ValueError(
            f"Found {(~valid).sum():,} "
            "invalid monthly transitions"
        )

    if (~ALLOWED[state, next_state]).any():
        raise ValueError(
            "Forbidden state transition found"
        )

    check = (
        (risk == 1)
        & (first != (next_state == 3))
    )

    if check.any():
        raise ValueError(
            "Inconsistent first90_next "
            "in eligible risk set"
        )

    return (
        state,
        next_state,
        history,
        risk,
        first,
        period,
        target_period,
    )


def fit_baselines():
    counts = np.zeros(
        (5, 8), dtype=np.int64
    )
    counts_hist = np.zeros(
        (5, 2, 8), dtype=np.int64
    )

    accepted = 0
    excluded = 0

    # New training period: 2006-2017.
    for split in ("early_train", "validation"):
        for frame in batches(split):
            s, y, h, _, _, _, target_month = parse(frame)

            keep = (
                target_month < TRAIN_TARGET_BEFORE
            )

            accepted += int(keep.sum())
            excluded += int((~keep).sum())

            counts += np.bincount(
                s[keep] * 8 + y[keep],
                minlength=5 * 8,
            ).reshape(5, 8)

            known = (
                keep & np.isin(h, [0, 1])
            )

            counts_hist += np.bincount(
                (
                    (s[known] * 2 + h[known])
                    * 8 + y[known]
                ),
                minlength=5 * 2 * 8,
            ).reshape(5, 2, 8)

    if (counts.sum(axis=1) == 0).any():
        raise ValueError(
            "A starting state has no "
            "training observations"
        )

    # Model 1: state-only historical frequencies.
    # Small Laplace smoothing prevents zero
    # probabilities for rare outcomes.
    parent = (
        counts + ALLOWED * 0.5
    ) / (
        counts.sum(axis=1, keepdims=True)
        + ALLOWED.sum(axis=1, keepdims=True) * 0.5
    )

    parent[~ALLOWED] = 0.0

    # Model 2: state + previous 90+ history.
    # Rare groups shrink toward state-only rates.
    hist = (
        counts_hist
        + SHRINKAGE * parent[:, None, :]
    ) / (
        counts_hist.sum(axis=2, keepdims=True)
        + SHRINKAGE
    )

    hist[
        ~np.broadcast_to(
            ALLOWED[:, None, :],
            hist.shape,
        )
    ] = 0.0

    return (
        parent,
        hist,
        counts,
        counts_hist,
        accepted,
        excluded,
    )


def evaluate(parent, hist):
    names = (
        "state_only",
        "state_plus_history",
    )

    metrics = {
        name: {
            "n": 0,
            "logloss": 0.0,
            "brier": 0.0,
            "observed": np.zeros(8),
            "expected": np.zeros(8),
            "first_n": 0,
            "first_obs": 0.0,
            "first_exp": 0.0,
            "by_state_n": np.zeros(5),
            "by_state_ll": np.zeros(5),
            "by_state_brier": np.zeros(5),
            "monthly": np.zeros((36, 6)),
        }
        for name in names
    }

    discarded = 0

    # New validation period: 2018-2020.
    # The 2021-2023 test set is never opened.
    for frame in batches("late_development"):
        s, y, h, risk, first, period, target = parse(
            frame
        )

        keep = (
            (period >= 201801)
            & (period <= 202012)
            & (target < VALID_TARGET_BEFORE)
        )

        discarded += int((~keep).sum())

        if not keep.any():
            continue

        s, y, h, risk, first, period = (
            v[keep]
            for v in (
                s, y, h, risk, first, period
            )
        )

        month_ix = (
            (period // 100 - 2018) * 12
            + (period % 100 - 1)
        )

        eligible = risk == 1

        for name in names:
            p = parent[s].copy()

            if name == "state_plus_history":
                known = np.isin(h, [0, 1])
                p[known] = hist[
                    s[known], h[known]
                ]

            selected = np.maximum(
                p[np.arange(len(y)), y],
                1e-15,
            )

            ll = -np.log(selected)

            bs = (
                1.0
                - 2.0 * selected
                + (p * p).sum(axis=1)
            )

            m = metrics[name]

            m["n"] += len(y)
            m["logloss"] += ll.sum()
            m["brier"] += bs.sum()

            m["observed"] += np.bincount(
                y, minlength=8
            )
            m["expected"] += p.sum(axis=0)

            m["first_n"] += int(
                eligible.sum()
            )
            m["first_obs"] += first[
                eligible
            ].sum()
            m["first_exp"] += p[
                eligible, 3
            ].sum()

            m["by_state_n"] += np.bincount(
                s, minlength=5
            )
            m["by_state_ll"] += np.bincount(
                s,
                weights=ll,
                minlength=5,
            )
            m["by_state_brier"] += np.bincount(
                s,
                weights=bs,
                minlength=5,
            )

            matrix = m["monthly"]

            # Monthly first-90 and credit-exit
            # calibration statistics.
            series = (
                (
                    eligible.astype(float),
                    first.astype(float) * eligible,
                    p[:, 3] * eligible,
                ),
                (
                    np.ones(len(y)),
                    (y == 6).astype(float),
                    p[:, 6],
                ),
            )

            for col, group in (
                (0, series[0]),
                (3, series[1]),
            ):
                for j, arr in enumerate(group):
                    matrix[:, col + j] += np.bincount(
                        month_ix,
                        weights=arr,
                        minlength=36,
                    )

    summary = []
    state_detail = []
    origin_detail = []
    monthly = []

    for name, m in metrics.items():
        if m["n"] == 0:
            raise ValueError(
                "No validation observations"
            )

        summary.append({
            "model": name,
            "validation_rows": int(m["n"]),
            "multiclass_logloss":
                m["logloss"] / m["n"],
            "multiclass_brier":
                m["brier"] / m["n"],
            "first90_risk_rows":
                int(m["first_n"]),
            "first90_actual":
                int(m["first_obs"]),
            "first90_expected":
                float(m["first_exp"]),
            "first90_OE":
                m["first_obs"] / m["first_exp"]
                if m["first_exp"] else np.nan,
            "credit_exit_OE":
                m["observed"][6] / m["expected"][6]
                if m["expected"][6] else np.nan,
            "payoff_OE":
                m["observed"][5] / m["expected"][5]
                if m["expected"][5] else np.nan,
        })

        for state in range(8):
            obs = m["observed"][state]
            exp = m["expected"][state]

            state_detail.append({
                "model": name,
                "next_state": f"S{state}",
                "actual": int(obs),
                "expected": float(exp),
                "OE": (
                    obs / exp if exp else np.nan
                ),
            })

        for state in range(5):
            n = m["by_state_n"][state]

            origin_detail.append({
                "model": name,
                "starting_state": f"S{state}",
                "rows": int(n),
                "logloss": (
                    m["by_state_ll"][state] / n
                    if n else np.nan
                ),
                "brier": (
                    m["by_state_brier"][state] / n
                    if n else np.nan
                ),
            })

        for i in range(36):
            mm = m["monthly"][i]

            monthly.append({
                "model": name,
                "period_yyyymm":
                    (2018 + i // 12) * 100
                    + (i % 12 + 1),
                "first90_risk_rows":
                    int(mm[0]),
                "first90_actual":
                    int(mm[1]),
                "first90_expected":
                    mm[2],
                "credit_rows":
                    int(mm[3]),
                "credit_actual":
                    int(mm[4]),
                "credit_expected":
                    mm[5],
            })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUT / "validation_summary.csv",
        index=False,
    )

    pd.DataFrame(state_detail).to_csv(
        OUT / "validation_by_next_state.csv",
        index=False,
    )

    pd.DataFrame(origin_detail).to_csv(
        OUT / "validation_by_start_state.csv",
        index=False,
    )

    pd.DataFrame(monthly).to_csv(
        OUT / "validation_monthly.csv",
        index=False,
    )

    return summary_df, discarded


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Task 29: Historical transition benchmarks",
        flush=True,
    )
    print(
        "Train: 2006-2017 | "
        "Validate: 2018-2020 | "
        "Test: untouched"
    )

    (
        parent,
        hist,
        counts,
        counts_hist,
        used,
        purged,
    ) = fit_baselines()

    np.savez_compressed(
        OUT / "transition_benchmarks.npz",
        state_only=parent,
        state_plus_history=hist,
        state_counts=counts,
        state_history_counts=counts_hist,
    )

    protocol = {
        "train_splits": [
            "early_train",
            "validation",
        ],
        "validation_split":
            "late_development",
        "test_split_not_accessed":
            "oot_test",
        "train_target_before":
            TRAIN_TARGET_BEFORE,
        "valid_target_before":
            VALID_TARGET_BEFORE,
        "history_shrinkage":
            SHRINKAGE,
        "target":
            "next_state_t1",
        "first90_defined_only_if_risk_set_1":
            True,
    }

    (OUT / "protocol.json").write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Training rows: {used:,}; "
        f"purged at cutoff: {purged:,}"
    )

    summary, excluded = evaluate(
        parent, hist
    )

    print(
        "Validation boundary-excluded rows: "
        f"{excluded:,}"
    )

    print(
        "\n=== VALIDATION RESULTS "
        "(NOT FINAL TEST) ==="
    )

    print(
        summary.round(5).to_string(
            index=False
        )
    )

    print(
        f"\nSaved outputs to {OUT}"
    )


if __name__ == "__main__":
    main()
