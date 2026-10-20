from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/30_full_multinomial_logistic.py'
OLD_SPEC = ROOT / 'configs/task30_feature_spec.json'
OUTBASE = ROOT / 'outputs/modeling/task31_6_state_lgbm'
DRIVEBASE = Path('/content/drive/MyDrive/creditport-lgbm-backup')

TRAIN_N = 24_023_978
VALID_N = 10_344_541
STOP_2018_N = 3_539_265

MACROS = {
    'hpi_yoy_pct',
    'unemployment_pct',
    'unemployment_change_3m_pp',
    'mortgage30_pct',
    'refinance_incentive_pp',
}

# Used ONLY to protect log calculations, not as predicted probability.
LOG_EPS = 1e-300


def import_source():
    spec = importlib.util.spec_from_file_location(
        'task30_for_lgbm', SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        json.dumps(obj, indent=2), encoding='utf-8'
    )
    os.replace(tmp, path)


def model_paths(folder, state):
    return (
        folder / f'S{state}.txt',
        folder / f'S{state}.json',
    )


def check_ram(limit, stage):
    rss = psutil.Process().memory_info().rss / 1024**3
    print(f'{stage}: RSS={rss:.1f} GiB', flush=True)
    if rss > limit:
        raise MemoryError(
            f'RSS exceeded {limit:.1f} GiB at {stage}'
        )


def build_features(frame, source, old, nums, cats):
    arr = np.empty(
        (len(frame), len(nums) + len(cats)),
        dtype=np.float32,
    )

    for j, name in enumerate(nums):
        arr[:, j] = source.numeric_array(
            frame[name]
        ).astype(np.float32)

    for j, name in enumerate(cats, start=len(nums)):
        levels = old['cat'][name]
        codes = pd.Categorical(
            source.cat_strings(frame[name]),
            categories=levels,
        ).codes

        arr[:, j] = np.where(
            codes < 0,
            levels.index('__UNSEEN__'),
            codes,
        )

    return arr


def collect_data(source, old, nums, cats, ram_limit):
    xt = [[] for _ in range(5)]
    yt = [[] for _ in range(5)]
    xv = [[] for _ in range(5)]
    yv = [[] for _ in range(5)]

    ntrain = 0
    nstop = 0

    print('\n=== CACHE TRAINING 2006-2017 ===', flush=True)

    for frame, states, y, _, _, _ in source.iterate_train():
        x = build_features(frame, source, old, nums, cats)

        for k in np.unique(states):
            mask = states == k
            xt[int(k)].append(x[mask].copy())
            yt[int(k)].append(y[mask].astype(np.int8))

        previous = ntrain
        ntrain += len(y)

        if ntrain // 2_000_000 != previous // 2_000_000:
            check_ram(ram_limit, f'Cached train {ntrain:,}')

    if ntrain != TRAIN_N:
        raise ValueError(
            f'Training rows {ntrain:,} != {TRAIN_N:,}'
        )

    print('\n=== CACHE 2018 EARLY STOP ===', flush=True)

    for raw in source.batches('late_development'):
        selected = source.select(raw, 'validate')

        if selected is None:
            continue

        frame, states, y, _, _, months = selected
        mask2018 = months // 100 == 2018

        if not mask2018.any():
            continue

        x = build_features(
            frame.loc[mask2018],
            source, old, nums, cats,
        )

        ss = states[mask2018]
        yy = y[mask2018]

        for k in np.unique(ss):
            mask = ss == k
            xv[int(k)].append(x[mask].copy())
            yv[int(k)].append(yy[mask].astype(np.int8))

        nstop += len(yy)

    if nstop != STOP_2018_N:
        raise ValueError(
            f'2018 rows {nstop:,} != {STOP_2018_N:,}'
        )

    check_ram(ram_limit, 'Cache complete')

    print(
        f'Rows verified: train={ntrain:,}, '
        f'stop2018={nstop:,}',
        flush=True,
    )

    return xt, yt, xv, yv


def full_label_feval(classes, true_labels):
    """
    Evaluate original S0-S7 labels, including labels
    absent from this state's training sample.
    """
    classes = np.asarray(classes)
    true_labels = np.asarray(true_labels)

    mapped = np.searchsorted(classes, true_labels)
    known = np.isin(true_labels, classes)

    ids = np.arange(len(true_labels))[known]
    cols = mapped[known]

    def evaluate(preds, _dataset):
        p = np.asarray(preds).reshape(
            len(true_labels), len(classes)
        )

        ptrue = np.zeros(
            len(true_labels), dtype=np.float64
        )
        ptrue[known] = p[ids, cols]

        loss = float(
            -np.log(np.maximum(ptrue, LOG_EPS)).mean()
        )

        return 'full_label_logloss', loss, False

    return evaluate


def fit_models(source, old, nums, cats, out, backup, args):
    todo = [
        k for k in range(5)
        if not all(
            p.is_file() for p in model_paths(out, k)
        )
    ]

    if not todo:
        print(
            'All five state models already saved.',
            flush=True,
        )
        return

    print('States to train:', todo, flush=True)

    xt, yt, xv, yv = collect_data(
        source, old, nums, cats, args.ram_stop_gib
    )

    names = nums + cats
    cat_positions = list(
        range(len(nums), len(names))
    )

    for k in todo:
        x = np.concatenate(xt[k])
        y = np.concatenate(yt[k])

        xt[k] = None
        yt[k] = None

        check_ram(
            args.ram_stop_gib, f'S{k} concatenated'
        )

        if len(y) < 10:
            raise ValueError(
                f'S{k}: fewer than 10 training rows'
            )

        if k == 4 and np.any(y == 3):
            raise ValueError(
                'Forbidden S4->S3 in training'
            )

        classes, counts = np.unique(
            y, return_counts=True
        )
        classes = classes.astype(np.int32)

        info = {
            'state': k,
            'type': (
                'lgbm' if len(classes) >= 2
                else 'constant'
            ),
            'classes': classes.tolist(),
            'train_rows': int(len(y)),
            'train_counts': {
                str(int(c)): int(n)
                for c, n in zip(classes, counts)
            },
        }

        print(
            f'\n=== S{k}: rows={len(y):,}; '
            f'targets={info["train_counts"]} ===',
            flush=True,
        )

        model_path, info_path = model_paths(out, k)

        if len(classes) == 1:
            tmp = model_path.with_suffix('.tmp')
            tmp.write_text(
                'CONSTANT_CLASS\n', encoding='utf-8'
            )
            os.replace(tmp, model_path)
            info['best_iteration'] = 0

        else:
            train_data = lgb.Dataset(
                x,
                label=np.searchsorted(classes, y),
                feature_name=names,
                categorical_feature=cat_positions,
                free_raw_data=True,
            )

            x = None
            y = None

            callbacks = [
                lgb.log_evaluation(period=0)
            ]
            valid_sets = []
            feval = None

            if xv[k]:
                vx = np.concatenate(xv[k])
                vy = np.concatenate(yv[k]).astype(
                    np.int32
                )

                mapped = np.searchsorted(classes, vy)
                known = np.isin(vy, classes)
                unseen_n = int((~known).sum())

                info['stop2018_unseen_class_rows'] = (
                    unseen_n
                )

                print(
                    f'S{k}: 2018 unseen target rows='
                    f'{unseen_n:,}',
                    flush=True,
                )

                # Placeholder labels only for LightGBM's
                # validation dataset. The custom metric
                # evaluates the true original labels.
                safe_labels = np.where(
                    known, mapped, 0
                )

                val_data = lgb.Dataset(
                    vx,
                    label=safe_labels,
                    reference=train_data,
                    feature_name=names,
                    categorical_feature=cat_positions,
                    free_raw_data=True,
                )

                valid_sets = [val_data]

                feval = full_label_feval(
                    classes, vy
                )

                callbacks.append(
                    lgb.early_stopping(
                        25,
                        first_metric_only=True,
                        verbose=False,
                    )
                )

            else:
                info['stop2018_unseen_class_rows'] = None
                print(
                    f'WARNING: S{k} has no 2018 '
                    'early-stopping observations.',
                    flush=True,
                )

            params = {
                'objective': 'multiclass',
                'num_class': len(classes),
                'metric': 'None',
                'learning_rate': 0.06,
                'num_leaves': 31,
                'max_depth': 7,
                'max_bin': 127,
                'min_data_in_leaf': max(
                    20,
                    min(
                        300,
                        info['train_rows'] // 1000,
                    ),
                ),
                'lambda_l2': 5.0,
                'num_threads': args.threads,
                'seed': 3106,
                'deterministic': True,
                'force_col_wise': True,
                'verbosity': -1,
            }

            booster = lgb.train(
                params,
                train_data,
                num_boost_round=args.rounds,
                valid_sets=valid_sets or None,
                feval=feval,
                callbacks=callbacks,
            )

            best = int(
                booster.best_iteration
                or booster.current_iteration()
            )
            info['best_iteration'] = best

            tmp = str(model_path) + '.tmp'

            booster.save_model(
                tmp, num_iteration=best
            )
            os.replace(tmp, model_path)

            del booster, train_data

            if valid_sets:
                del val_data, vx, vy

        write_json(info_path, info)

        shutil.copy2(
            model_path,
            backup / model_path.name,
        )
        shutil.copy2(
            info_path,
            backup / info_path.name,
        )

        xv[k] = None
        yv[k] = None

        print(
            f'S{k}: saved + Drive backup; '
            f'best_round={info["best_iteration"]}',
            flush=True,
        )


def load_models(out):
    models = {}

    for k in range(5):
        model_path, info_path = model_paths(out, k)

        info = json.loads(
            info_path.read_text(encoding='utf-8')
        )

        if info['state'] != k:
            raise ValueError(
                f'S{k}: metadata state mismatch'
            )

        booster = (
            lgb.Booster(model_file=str(model_path))
            if info['type'] == 'lgbm'
            else None
        )

        models[k] = (booster, info)

    return models


def predict_states(x, states, models, threads):
    p = np.zeros(
        (len(states), 8), dtype=np.float64
    )

    for k in range(5):
        ids = np.flatnonzero(states == k)

        if len(ids) == 0:
            continue

        booster, info = models[k]
        classes = np.asarray(
            info['classes'], dtype=np.int32
        )

        if booster is None:
            p[ids, classes[0]] = 1.0

        else:
            pred = np.asarray(
                booster.predict(
                    x[ids], num_threads=threads
                )
            )

            pred = pred.reshape(
                len(ids), len(classes)
            )

            p[np.ix_(ids, classes)] = pred

    # No invented probability for unseen classes.
    if (
        not np.isfinite(p).all()
        or np.any(p < -1e-12)
    ):
        raise ValueError(
            'Invalid LightGBM probabilities'
        )

    if np.max(
        np.abs(p.sum(axis=1) - 1.0)
    ) > 1e-6:
        raise ValueError(
            'Probability rows do not sum to 1'
        )

    if np.any(
        (states == 4) & (p[:, 3] != 0)
    ):
        raise ValueError(
            'Forbidden S4->S3 probability is positive'
        )

    return p


def validate(source, old, nums, cats, out, threads):
    models = load_models(out)

    buckets = defaultdict(
        lambda: {
            'rows': 0,
            'll': 0.0,
            'bs': 0.0,
            'actual': np.zeros(8),
            'expected': np.zeros(8),
            'first_actual': 0.0,
            'first_expected': 0.0,
        }
    )
    unseen = defaultdict(int)

    def add(group, y, p, risk, first):
        b = buckets[group]
        n = len(y)

        if not n:
            return

        ptrue = p[np.arange(n), y]

        b['rows'] += n
        b['ll'] += float(
            -np.log(
                np.maximum(ptrue, LOG_EPS)
            ).sum()
        )
        b['bs'] += float(
            (
                1 - 2 * ptrue
                + (p * p).sum(axis=1)
            ).sum()
        )

        b['actual'] += np.bincount(
            y, minlength=8
        )
        b['expected'] += p.sum(axis=0)

        eligible = risk == 1

        b['first_actual'] += float(
            first[eligible].sum()
        )
        b['first_expected'] += float(
            p[eligible, 3].sum()
        )

    print(
        '\n=== VALIDATE DEVELOPMENT 2018-2020 ===',
        flush=True,
    )

    for raw in source.batches('late_development'):
        item = source.select(raw, 'validate')

        if item is None:
            continue

        frame, states, y, risk, first, months = item

        x = build_features(
            frame, source, old, nums, cats
        )
        p = predict_states(
            x, states, models, threads
        )

        add('all', y, p, risk, first)

        for k in np.unique(states):
            mask = states == k

            add(
                f'start_S{k}',
                y[mask], p[mask],
                risk[mask], first[mask],
            )

            classes = models[int(k)][1]['classes']

            unseen[int(k)] += int(
                (~np.isin(
                    y[mask], classes
                )).sum()
            )

        years = months // 100

        for yr in np.unique(years):
            mask = years == yr

            add(
                f'year_{yr}',
                y[mask], p[mask],
                risk[mask], first[mask],
            )

    if buckets['all']['rows'] != VALID_N:
        raise ValueError(
            f'Validation rows '
            f'{buckets["all"]["rows"]:,} '
            f'!= {VALID_N:,}'
        )

    def oe(a, e):
        return (
            float(a / e)
            if e > 0
            else np.nan
        )

    table = []

    for group, b in sorted(buckets.items()):
        r = {
            'model': out.name,
            'group': group,
            'rows': b['rows'],
            'logloss': b['ll'] / b['rows'],
            'brier': b['bs'] / b['rows'],
            'first90_actual': int(
                b['first_actual']
            ),
            'first90_expected': b['first_expected'],
            'first90_OE': oe(
                b['first_actual'],
                b['first_expected'],
            ),
        }

        for state in range(8):
            r[f'S{state}_actual'] = int(
                b['actual'][state]
            )
            r[f'S{state}_expected'] = float(
                b['expected'][state]
            )
            r[f'S{state}_OE'] = oe(
                b['actual'][state],
                b['expected'][state],
            )

        table.append(r)

    result = pd.DataFrame(table)

    result.to_csv(
        out / 'validation_breakdown.csv',
        index=False,
    )

    diagnostic = {
        'unseen_validation_labels_by_start_state': (
            dict(unseen)
        ),
        'total_unseen_validation_labels': int(
            sum(unseen.values())
        ),
        'note': (
            'Absent training classes have probability 0; '
            'logloss clipping is numerical only.'
        ),
    }

    write_json(
        out / 'validation_diagnostics.json',
        diagnostic,
    )

    selected = [
        'all',
        'year_2018',
        'year_2019',
        'year_2020',
    ]

    view = result[
        result['group'].isin(selected)
    ]

    print(
        view[
            [
                'group',
                'rows',
                'logloss',
                'brier',
                'first90_OE',
                'S5_OE',
                'S6_OE',
            ]
        ].round(5).to_string(index=False),
        flush=True,
    )

    print(
        'Unseen target events by starting state:',
        dict(unseen),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--macro',
        choices=['off', 'on'],
        default='off',
    )
    ap.add_argument(
        '--rounds',
        type=int,
        default=180,
    )
    ap.add_argument(
        '--threads',
        type=int,
        default=8,
    )
    ap.add_argument(
        '--ram-stop-gib',
        type=float,
        default=38,
    )

    args = ap.parse_args()

    if args.rounds < 10 or args.threads < 1:
        ap.error(
            'Require rounds>=10 and threads>=1'
        )

    if not Path(
        '/content/drive/MyDrive'
    ).is_dir():
        raise RuntimeError(
            'Mount Google Drive first'
        )

    os.environ['CREDIT_PORT_ROOT'] = str(ROOT)

    source = import_source()

    if (
        source.TRAIN_TARGET_BEFORE != 201712
        or source.VALID_TARGET_BEFORE != 202012
    ):
        raise ValueError(
            'Task 30 time boundaries changed'
        )

    old = json.loads(
        OLD_SPEC.read_text(encoding='utf-8')
    )

    if int(old['rows']) != TRAIN_N:
        raise ValueError(
            'Task 30 feature_spec '
            'training count mismatch'
        )

    nums = [
        x for x in source.NUM
        if x in old['num']
        and (
            args.macro == 'on'
            or x not in MACROS
        )
    ]

    cats = [
        x for x in source.CAT
        if x != 'state_t'
    ]

    for c in cats:
        if (
            c not in old['cat']
            or '__UNSEEN__' not in old['cat'][c]
        ):
            raise ValueError(
                f'Missing Task 30 categorical '
                f'specification: {c}'
            )

    tag = (
        'withmacro'
        if args.macro == 'on'
        else 'nomacro'
    )

    out = OUTBASE / tag
    backup = DRIVEBASE / tag

    out.mkdir(
        parents=True, exist_ok=True
    )
    backup.mkdir(
        parents=True, exist_ok=True
    )

    manifest = {
        'implementation': (
            'reviewed_v2_full_label_stop_'
            'no_probability_floor'
        ),
        'macro': args.macro,
        'nums': nums,
        'cats': cats,
        'old_spec_sha256': hashlib.sha256(
            OLD_SPEC.read_bytes()
        ).hexdigest(),
        'rounds': args.rounds,
        'seed': 3106,
        'training_rows': TRAIN_N,
        'stop_year': 2018,
        'evaluation_years': [
            2018, 2019, 2020
        ],
    }

    for path in (
        out / 'manifest.json',
        backup / 'manifest.json',
    ):
        if path.exists():
            existing = json.loads(
                path.read_text(encoding='utf-8')
            )
            if existing != manifest:
                raise ValueError(
                    f'Incompatible manifest: {path}'
                )

    write_json(
        out / 'manifest.json',
        manifest,
    )
    shutil.copy2(
        out / 'manifest.json',
        backup / 'manifest.json',
    )

    for k in range(5):
        local_pair = model_paths(out, k)
        remote_pair = model_paths(backup, k)

        if (
            not all(
                p.is_file() for p in local_pair
            )
            and all(
                p.is_file() for p in remote_pair
            )
        ):
            for remote, local in zip(
                remote_pair, local_pair
            ):
                shutil.copy2(
                    remote, local
                )

            print(
                f'Restored S{k} from Drive',
                flush=True,
            )

    t = time.perf_counter()

    print(
        f'Task31.6 | macro={args.macro} | '
        f'features={len(nums) + len(cats)}',
        flush=True,
    )

    fit_models(
        source, old, nums, cats,
        out, backup, args,
    )

    validate(
        source, old, nums, cats,
        out, args.threads,
    )

    for name in [
        'validation_breakdown.csv',
        'validation_diagnostics.json',
    ]:
        shutil.copy2(
            out / name,
            backup / name,
        )

    print(
        f'Done in '
        f'{(time.perf_counter() - t)/60:.1f} min; '
        f'backup={backup}',
        flush=True,
    )

    print(
        '2021-2023 OOT test was NOT accessed.',
        flush=True,
    )


if __name__ == '__main__':
    main()
