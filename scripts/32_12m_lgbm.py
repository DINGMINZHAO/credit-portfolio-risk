from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/model_ready/28_monthly_transitions/late_development'
MODELS = ROOT / 'models'
SPEC = ROOT / 'configs/task30_feature_spec.json'
OUT = ROOT / 'outputs/modeling/task32_12m'


def load_original(variant):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import importlib.util

    path = ROOT / 'scripts/31_6_state_specific_lgbm.py'
    sp = importlib.util.spec_from_file_location('original_lgbm', path)
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)

    source = module.import_source()
    old = json.loads(SPEC.read_text())
    man = json.loads((MODELS / variant / 'manifest.json').read_text())

    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    if digest != man['old_spec_sha256'] or old['rows'] != 24_023_978:
        raise RuntimeError('Original Task 30 feature specification mismatch')

    models = module.load_models(MODELS / variant)

    for state in (0, 1, 2):
        if models[state][1]['classes'] != list(range(8)):
            raise RuntimeError(f'Unexpected S{state} output classes')
        if models[state][0].feature_name() != man['nums'] + man['cats']:
            raise RuntimeError(f'Unexpected S{state} model features')

    print('PASS: exact original feature mapping and saved models', flush=True)
    return module, source, old, man, models


def load_snapshot(n, seed):
    cols = [
        'loan_id', 'period_yyyymm', 'state_t', 'first90_risk_set',
        'orig_credit_score', 'orig_ltv', 'orig_dti', 'log_orig_upb',
        'orig_term_months', 'loan_age_reported_t',
        'months_remaining_reported_t', 'log_upb_t', 'upb_t',
        'interest_rate_pct_t', 'orig_interest_rate_pct',
        'months_90plus_observed_through_t',
        'months_delinquent_observed_through_t',
        'hpi_yoy_pct', 'unemployment_pct',
        'unemployment_change_3m_pp', 'mortgage30_pct',
        'refinance_incentive_pp', 'prior_observed_state',
        'ever_90plus_through_t', 'orig_property_state',
        'orig_occupancy', 'orig_loan_purpose', 'modification_flag_t',
    ]

    parts = []
    for path in sorted(DATA.glob('*.parquet')):
        table = pq.read_table(
            path, columns=cols,
            filters=[('period_yyyymm', '=', 201901)],
        )

        if table.num_rows == 0:
            continue

        f = table.to_pandas()
        s = pd.to_numeric(f['state_t'], errors='coerce')
        risk = pd.to_numeric(f['first90_risk_set'], errors='coerce')
        f = f.loc[(risk == 1) & s.isin([0, 1, 2])].copy()

        if len(f):
            f['_vintage_file'] = path.name
            parts.append(f)

    if not parts:
        raise RuntimeError('No eligible 2019-01 development snapshots')

    all_snapshots = pd.concat(parts, ignore_index=True)

    if all_snapshots.duplicated(['_vintage_file', 'loan_id']).any():
        raise RuntimeError('Duplicate loan IDs within vintage')

    total = len(all_snapshots)
    selected = all_snapshots.sample(
        n=min(n, total), random_state=seed
    )

    print(
        f'2019-01 risk set: {total:,}; sampled {len(selected):,}',
        flush=True
    )
    return selected.reset_index(drop=True), total


def simulate(snapshot, module, source, old, man, models, particles, seed):
    n = len(snapshot)
    x = snapshot.loc[
        snapshot.index.repeat(particles)
    ].reset_index(drop=True).copy()

    m = len(x)
    rng = np.random.default_rng(seed)

    state = pd.to_numeric(x['state_t']).to_numpy(dtype=np.int8)
    balance = pd.to_numeric(
        x['upb_t'], errors='coerce'
    ).to_numpy(dtype=float)

    if not np.isfinite(balance).all() or (balance < 0).any():
        raise RuntimeError('Invalid starting UPB')

    initial_log = pd.to_numeric(
        x['log_upb_t'], errors='coerce'
    ).to_numpy(dtype=float)

    if not np.allclose(
        initial_log, np.log1p(balance),
        atol=1e-4, rtol=1e-4
    ):
        raise RuntimeError('log_upb_t transformation mismatch')

    rate = pd.to_numeric(
        x['interest_rate_pct_t'], errors='coerce'
    )
    fallback = pd.to_numeric(
        x['orig_interest_rate_pct'], errors='coerce'
    )
    annual_rate = rate.fillna(fallback).to_numpy(dtype=float)

    if not np.isfinite(annual_rate).all():
        raise RuntimeError('Missing mortgage rate')

    monthly_rate = np.clip(annual_rate / 1200.0, 0, 0.05)

    age = pd.to_numeric(
        x['loan_age_reported_t'], errors='coerce'
    ).to_numpy(dtype=float)

    remaining = pd.to_numeric(
        x['months_remaining_reported_t'], errors='coerce'
    ).to_numpy(dtype=float)

    past_delinq = pd.to_numeric(
        x['months_delinquent_observed_through_t'],
        errors='coerce'
    ).to_numpy(dtype=float)

    weight = np.ones(m, dtype=float)
    event = np.zeros(m, dtype=float)
    exits = np.zeros(m, dtype=float)
    pd_by_month = []

    for h in range(1, 13):
        features = module.build_features(
            x, source, old, man['nums'], man['cats']
        )
        p = module.predict_states(
            features, state, models, threads=4
        )

        hazard = p[:, 3]
        exitprob = p[:, 4:].sum(axis=1)
        survival = p[:, :3].sum(axis=1)

        event += weight * hazard
        exits += weight * exitprob
        weight *= survival

        pd_by_month.append(
            event.reshape(n, particles).mean(axis=1)
        )

        print(
            f'Month {h:02d}/12: '
            f'mean cumulative PD={pd_by_month[-1].mean():.4%}',
            flush=True
        )

        if h == 12:
            break

        # Simulate conditional surviving states S0/S1/S2.
        draw = rng.random(m) * survival
        nxt = np.where(
            draw < p[:, 0], 0,
            np.where(draw < p[:, 0] + p[:, 1], 1, 2)
        ).astype(np.int8)

        previous = state.copy()
        state = nxt

        x['prior_observed_state'] = pd.Series(
            previous, index=x.index
        ).astype(str)
        x['state_t'] = state

        # Update dynamic loan features.
        age = age + 1
        remaining = np.maximum(remaining - 1, 0)
        past_delinq = past_delinq + np.isin(state, [1, 2])

        x['loan_age_reported_t'] = age
        x['months_remaining_reported_t'] = remaining
        x['months_delinquent_observed_through_t'] = past_delinq

        # First90 has not happened on surviving paths.
        # No new modification is assumed.
        # Future macro feature values remain fixed at as-of levels.

        # Scheduled amortization approximation:
        # S0 amortizes; S1/S2 balances remain unchanged.
        terms = np.maximum(remaining + 1, 1)
        denom = -np.expm1(
            -terms * np.log1p(monthly_rate)
        )

        installment = np.divide(
            balance * monthly_rate, denom,
            out=np.zeros_like(balance),
            where=denom > 1e-12,
        )

        installment = np.where(
            monthly_rate < 1e-12,
            balance / terms,
            installment
        )

        principal = np.clip(
            installment - balance * monthly_rate,
            0, balance
        )

        balance = np.where(
            state == 0, balance - principal, balance
        )

        x['upb_t'] = balance
        x['log_upb_t'] = np.log1p(balance)

    pd12 = event.reshape(n, particles).mean(axis=1)
    exit12 = exits.reshape(n, particles).mean(axis=1)
    active12 = weight.reshape(n, particles).mean(axis=1)

    if (
        np.abs(pd12 + exit12 + active12 - 1) > 1e-6
    ).any():
        raise RuntimeError('Probability mass not conserved')

    curve = np.stack(pd_by_month)

    if (np.diff(curve, axis=0) < -1e-10).any():
        raise RuntimeError('Cumulative PD not monotone')

    return pd12, exit12, active12


def label_snapshot(sample):
    # Only 2019 development records are read.
    # December 2019 transitions target January 2020.
    cols = [
        'loan_id', 'period_yyyymm', 'target_period_yyyymm',
        'state_t', 'next_state_t1', 'first90_next',
    ]

    observed = {}

    for fname, group in sample.groupby('_vintage_file'):
        ids = set(group['loan_id'].astype(str))

        table = pq.read_table(
            DATA / fname, columns=cols,
            filters=[
                ('period_yyyymm', '>=', 201901),
                ('period_yyyymm', '<=', 201912),
            ],
        )

        frame = table.to_pandas()
        frame = frame.loc[
            frame['loan_id'].astype(str).isin(ids)
        ]

        for lid, hist in frame.groupby('loan_id'):
            by_month = {
                int(r.period_yyyymm): r
                for r in hist.itertuples(index=False)
            }

            outcome = np.nan
            last_state = None

            for h in range(12):
                month = 201901 + h
                next_month = (
                    201902 + h if h < 11 else 202001
                )

                r = by_month.get(month)

                if (
                    r is None
                    or int(r.target_period_yyyymm) != next_month
                ):
                    break

                if (
                    last_state is not None
                    and int(r.state_t) != last_state
                ):
                    break

                nxt = int(r.next_state_t1)

                if nxt == 3:
                    if int(r.first90_next) != 1:
                        raise RuntimeError('First90 label mismatch')
                    outcome = 1
                    break

                if nxt in (4, 5, 6, 7):
                    outcome = 0
                    break

                if nxt not in (0, 1, 2):
                    break

                if h == 11:
                    outcome = 0

                last_state = nxt

            observed[(fname, str(lid))] = outcome

        print(
            f'Labels scanned: {fname}; '
            f'sampled loans={len(ids)}',
            flush=True
        )

    return [
        observed.get((fname, str(lid)), np.nan)
        for fname, lid in sample[
            ['_vintage_file', 'loan_id']
        ].itertuples(index=False, name=None)
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--variant', default='withmacro',
        choices=['withmacro', 'nomacro']
    )
    ap.add_argument('--loans', default=1000, type=int)
    ap.add_argument('--particles', default=64, type=int)

    args = ap.parse_args()

    if args.loans < 1 or args.particles < 2:
        raise ValueError('Invalid loan or particle count')

    module, source, old, man, models = load_original(
        args.variant
    )

    snapshot, risk_set_n = load_snapshot(
        args.loans, seed=3201
    )

    pd12, exits, alive = simulate(
        snapshot, module, source, old, man, models,
        particles=args.particles, seed=3202,
    )

    result = snapshot[
        ['_vintage_file', 'loan_id', 'state_t']
    ].copy()

    result['asof_yyyymm'] = 201901
    result['pd_12m'] = pd12
    result['exit_before_first90_12m'] = exits
    result['active_no_first90_12m'] = alive
    result['first90_12m_actual'] = label_snapshot(snapshot)

    valid = result['first90_12m_actual'].notna()
    evaldf = result.loc[valid]

    y = evaldf['first90_12m_actual'].to_numpy(dtype=float)
    p = evaldf['pd_12m'].to_numpy(dtype=float)

    print('\n=== TASK 32 | 12M PD | 2019-01 SNAPSHOT ===')
    print(f'Model: {args.variant}; Flat-feature macro scenario')
    print(
        f'Portfolio risk set: {risk_set_n:,}; '
        f'sample: {len(result):,}'
    )
    print(
        f'Mature labels: {len(evaldf):,}; '
        f'censored: {(~valid).sum():,}'
    )

    if len(evaldf):
        print(
            f'Observed First90: {y.sum():,.0f}; '
            f'Expected: {p.sum():,.2f}'
        )
        print(
            f'O/E: {y.sum()/p.sum():.4f}'
            if p.sum() else 'O/E undefined'
        )
        print(f'Brier: {np.mean((y-p)**2):.6f}')

        q = np.clip(p, 1e-15, 1 - 1e-15)
        ll = -np.mean(
            y * np.log(q)
            + (1-y) * np.log1p(-q)
        )
        print(f'LogLoss: {ll:.6f}')

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'12m_lgbm_{args.variant}_201901_sample.csv'
    result.to_csv(path, index=False)

    print(f'Saved: {path}')
    print('No OOT accessed; no model training.')
    print('MVP assumptions: no new modifications;')
    print('scheduled amortization for S0; flat UPB for S1/S2;')
    print('future macro features fixed at as-of values.')


if __name__ == '__main__':
    main()
