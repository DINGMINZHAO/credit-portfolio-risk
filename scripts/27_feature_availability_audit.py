
"""Task 27.4A: full feature availability/quality audit; does not change data."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.cwd()
RAW = ROOT / 'data/raw'
BASE = ROOT / 'data/processed/monthly_model_base'
OUT = ROOT / 'outputs/pd_model'
YEARS = range(2006, 2023)
CHUNK = 150_000

# 0-based column indices: Freddie Mac origination/performance format.
ORIG = {
    'credit_score': (0, 'num', 9999, 300, 850),
    'first_payment_date': (1, 'date', None, None, None),
    'first_time_homebuyer': (2, 'cat', '9', None, None),
    'msa': (4, 'cat', '', None, None),
    'mi_pct': (5, 'num', 999, 0, 55),
    'number_units': (6, 'num', 99, 1, 4),
    'occupancy': (7, 'cat', '9', None, None),
    'original_cltv': (8, 'num', 999, 1, 998),
    'original_dti': (9, 'num', 999, 0, 65),
    'original_upb': (10, 'num', None, 1, None),
    'original_ltv': (11, 'num', 999, 1, 998),
    'original_rate_pct': (12, 'num', None, 0.001, 30),
    'origination_channel': (13, 'cat', '9', None, None),
    'prepayment_penalty': (14, 'cat', None, None, None),
    'amortization_type': (15, 'cat', None, None, None),
    'property_state': (16, 'cat', None, None, None),
    'property_type': (17, 'cat', '99', None, None),
    'postal_code': (18, 'cat', '00000', None, None),
    'loan_purpose': (20, 'cat', '9', None, None),
    'original_term': (21, 'num', None, 1, 600),
    'borrowers': (22, 'num', 99, 1, 10),
    'special_program': (27, 'cat', None, None, None),
    'relief_refinance': (28, 'cat', None, None, None),
    'valuation_method': (29, 'cat', '7', None, None),
    'interest_only': (30, 'cat', None, None, None),
    # Origination col 32: MI cancellation is updated after acquisition;
    # not a point-in-time origination feature. Inspect only for leakage.
}

PERF = {
    'step_rate_flag': (23, 'flag'),
    'payment_deferral_flag': (24, 'flag'),
    'estimated_ltv': (25, 'eltv'),
    'disaster_delinquency': (28, 'flag'),
    'borrower_assistance': (29, 'flag'),
    'interest_bearing_upb': (31, 'amount'),
}

MONTHLY_NUM = [
    'upb_t', 'loan_age_reported_t',
    'months_remaining_reported_t', 'interest_rate_pct_t',
    'history_gap_count_through_t', 'observed_months_through_t',
    'months_since_first_observation_t',
    'months_90plus_observed_through_t',
    'months_delinquent_observed_through_t',
]

MONTHLY_CAT = [
    'modification_flag_t', 'prior_observed_state',
    'ever_90plus_through_t', 'history_gap_before_t'
]


def member(z, kind, year):
    name = f'sample_{kind}_{year}.txt'
    matches = [
        v for v in z.namelist()
        if Path(v).name.lower() == name
    ]
    if len(matches) != 1:
        raise ValueError(f'Missing or duplicate {name}: {matches}')
    return matches[0]


def bucket(yyyymm):
    y = np.asarray(yyyymm, dtype=np.int64) // 100
    return np.select(
        [y <= 2014, y <= 2020, y <= 2023],
        ['2006-2014', '2015-2020', '2021-2023'],
        default='2024-2026'
    )


def summarize(field, values, typ, missing_code=None, lo=None, hi=None):
    s = pd.Series(values, copy=False)
    raw = s.astype('string').str.strip()
    empty = raw.isna() | raw.eq('')

    if typ in ('num', 'date', 'eltv', 'amount'):
        a = pd.to_numeric(
            raw, errors='coerce'
        ).to_numpy(dtype='float64', na_value=np.nan)

        sentinel = np.zeros(len(a), dtype=bool)
        if missing_code is not None:
            sentinel = np.isfinite(a) & (a == float(missing_code))

        valid = np.isfinite(a) & ~sentinel

        if typ == 'date':
            int_date = np.where(
                np.isfinite(a), a, 0
            ).astype(np.int64)
            bad = valid & (
                (int_date % 100 < 1) |
                (int_date % 100 > 12) |
                (int_date // 100 < 1900) |
                (int_date // 100 > 2050)
            )
        else:
            bad = np.zeros(len(a), dtype=bool)
            if lo is not None:
                bad |= valid & (a < lo)
            if hi is not None:
                bad |= valid & (a > hi)

        usable = valid & ~bad
        present = int(usable.sum())

        return dict(
            total=len(a),
            blank=int(empty.sum()),
            sentinel=int(sentinel.sum()),
            usable=present,
            out_of_range=int(bad.sum()),
            min_value=float(a[usable].min()) if present else None,
            max_value=float(a[usable].max()) if present else None,
            nonempty=int((~empty).sum()),
            distinct=None
        )

    sentinel = (
        raw.eq(str(missing_code)).fillna(False)
        if missing_code is not None
        else np.zeros(len(raw), dtype=bool)
    )
    valid = (~empty) & ~sentinel

    return dict(
        total=len(raw),
        blank=int(empty.sum()),
        sentinel=int(np.asarray(sentinel).sum()),
        usable=int(valid.sum()),
        out_of_range=0,
        min_value=None,
        max_value=None,
        nonempty=int((~empty).sum()),
        distinct=int(raw[valid].nunique())
    )


def add_result(records, source, year, field, typ,
               values, missing_code=None, lo=None, hi=None,
               cohort=None):
    d = summarize(field, values, typ, missing_code, lo, hi)
    d.update(
        source=source,
        vintage=year,
        feature=field,
        period=cohort or 'ALL'
    )
    records.append(d)


def read_orig(year, records, checks):
    with ZipFile(RAW / f'sample_{year}.zip') as z:
        m = member(z, 'orig', year)
        with z.open(m) as f:
            first = f.readline().decode('utf-8').rstrip('\r\n')

        width = len(first.split('|'))
        if width not in (31, 32):
            raise ValueError(
                f'{year}: unexpected origination width {width}'
            )

        with z.open(m) as f:
            df = pd.read_csv(
                f, sep='|', header=None,
                dtype=str, keep_default_na=False,
                encoding='utf-8'
            )

    checks[(year, 'orig_columns')] = width
    checks[(year, 'orig_rows')] = len(df)

    for name, (pos, typ, missing, lo, hi) in ORIG.items():
        if pos >= width:
            continue
        add_result(
            records, 'origination_all_loans', year,
            name, typ, df.iloc[:, pos], missing, lo, hi
        )

    checks[(year, 'orig_contains_last_servicer')] = int(width > 24)
    checks[(year, 'orig_contains_mi_cancellation')] = int(width >= 32)


def monthly_audit(year, records, checks, cohorts):
    path = BASE / f'27_monthly_base_{year}.parquet'
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)

    fields = [
        x for x in MONTHLY_NUM + MONTHLY_CAT
        if x in available
    ]

    required = {
        'period_yyyymm', 'state_t',
        'label_eligible', 'next_state_t1',
        'ever_90plus_through_t',
        'months_90plus_observed_through_t',
        'observed_months_through_t'
    }

    if not required <= available:
        raise ValueError(
            f'{path}: missing fields {required - available}'
        )

    readcols = sorted(set(fields) | required)
    agg = defaultdict(Counter)
    minima, maxima = {}, {}
    cat_sets = defaultdict(set)

    rows, eligible = 0, 0

    for batch in pf.iter_batches(
        batch_size=CHUNK, columns=readcols
    ):
        frame = batch.to_pandas()
        rows += len(frame)

        state = frame['state_t'].to_numpy()
        active = (state >= 0) & (state <= 4)
        label = frame['label_eligible'].to_numpy(dtype=bool)
        eligible += int(label.sum())

        p = frame['period_yyyymm'].to_numpy()
        period_group = bucket(p)

        for field in fields:
            s = frame[field]

            if field in MONTHLY_NUM:
                a = pd.to_numeric(
                    s, errors='coerce'
                ).to_numpy(dtype='float64', na_value=np.nan)

                good = active & np.isfinite(a)
                bad = active & ~np.isfinite(a)
                outlier = good & (a < 0)
                good &= ~outlier

                for group in np.unique(period_group):
                    ix = active & (period_group == group)
                    key = (field, str(group))

                    agg[key]['total'] += int(ix.sum())
                    agg[key]['usable'] += int((ix & good).sum())
                    agg[key]['blank'] += int((ix & bad).sum())
                    agg[key]['out_of_range'] += int(
                        (ix & outlier).sum()
                    )

                    values = a[ix & good]
                    if len(values):
                        minima[key] = min(
                            minima.get(key, np.inf),
                            float(values.min())
                        )
                        maxima[key] = max(
                            maxima.get(key, -np.inf),
                            float(values.max())
                        )
            else:
                sv = s.astype('string').str.strip()
                nonempty = (~sv.isna()) & sv.ne('')

                for group in np.unique(period_group):
                    ix = active & (period_group == group)
                    key = (field, str(group))

                    agg[key]['total'] += int(ix.sum())
                    agg[key]['usable'] += int(
                        (ix & nonempty).sum()
                    )
                    agg[key]['blank'] += int(
                        (ix & ~nonempty).sum()
                    )
                    cat_sets[key].update(
                        sv[ix & nonempty].unique().tolist()
                    )

        e = frame['ever_90plus_through_t'].fillna(
            False
        ).to_numpy(dtype=bool)

        c = frame[
            'months_90plus_observed_through_t'
        ].to_numpy()

        o = frame[
            'observed_months_through_t'
        ].to_numpy()

        checks[(year, 'history_90_flag_inconsistent')] += int(
            (e != (c > 0)).sum()
        )
        checks[(year, 'history_count_exceeds_observations')] += int(
            (c > o).sum()
        )
        checks[(year, 'eligible_terminal_origin')] += int(
            (label & (state >= 5)).sum()
        )
        checks[(year, 'eligible_unknown_destination')] += int(
            (
                label &
                (frame['next_state_t1'].to_numpy() < 0)
            ).sum()
        )

        for cohort in np.unique(period_group):
            ix = active & (period_group == cohort)
            cohorts[
                (str(cohort), 'active_monthly_rows')
            ] += int(ix.sum())
            cohorts[
                (str(cohort), 'labels')
            ] += int((ix & label).sum())

    checks[(year, 'monthly_rows')] = rows
    checks[(year, 'valid_labels')] = eligible

    for (field, group), ctr in agg.items():
        records.append(dict(
            source='monthly_base_active_rows',
            vintage=year,
            period=group,
            feature=field,
            total=ctr['total'],
            usable=ctr['usable'],
            blank=ctr['blank'],
            sentinel=0,
            out_of_range=ctr['out_of_range'],
            nonempty=ctr['usable'],
            distinct=(
                len(cat_sets[(field, group)])
                if field in MONTHLY_CAT else None
            ),
            min_value=minima.get((field, group)),
            max_value=maxima.get((field, group))
        ))


def read_perf_extras(year, records, checks, cohorts):
    with ZipFile(RAW / f'sample_{year}.zip') as z:
        m = member(z, 'perf', year)

        with z.open(m) as f:
            width = len(
                f.readline().decode('utf-8')
                .rstrip('\r\n').split('|')
            )

        checks[(year, 'performance_columns')] = width

        positions = {
            name: pos for name, (pos, _) in PERF.items()
            if pos < width
        }

        cols = sorted({1, 8} | set(positions.values()))
        totals = defaultdict(Counter)
        mins, maxs = {}, {}

        with z.open(m) as f:
            reader = pd.read_csv(
                f,
                sep='|',
                header=None,
                usecols=cols,
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK,
                encoding='utf-8'
            )

            for chunk in reader:
                checks[
                    (year, 'performance_rows')
                ] += len(chunk)

                period = pd.to_numeric(
                    chunk[1], errors='coerce'
                ).fillna(-1).to_numpy(dtype=np.int64)

                groups = bucket(period)
                active = chunk[8].str.strip().eq('').to_numpy()

                for feature, pos in positions.items():
                    typ = PERF[feature][1]
                    raw = chunk[pos].astype('string').str.strip()

                    arr = (
                        pd.to_numeric(
                            raw, errors='coerce'
                        ).to_numpy(dtype=float, na_value=np.nan)
                        if typ in ('eltv', 'amount')
                        else None
                    )

                    if typ == 'eltv':
                        valid = (
                            np.isfinite(arr) &
                            (arr != 999) &
                            (arr >= 1) &
                            (arr <= 998)
                        )
                        sentinel = (
                            np.isfinite(arr) & (arr == 999)
                        )
                        bad = (
                            np.isfinite(arr) &
                            (arr != 999) & ~valid
                        )

                    elif typ == 'amount':
                        valid = (
                            np.isfinite(arr) & (arr >= 0)
                        )
                        sentinel = np.zeros(
                            len(chunk), dtype=bool
                        )
                        bad = (
                            np.isfinite(arr) & (arr < 0)
                        )

                    else:
                        valid = raw.ne('').fillna(
                            False
                        ).to_numpy(dtype=bool)
                        sentinel = np.zeros(
                            len(chunk), dtype=bool
                        )
                        bad = np.zeros(
                            len(chunk), dtype=bool
                        )

                    for group in np.unique(groups):
                        mask = active & (groups == group)
                        key = (feature, str(group))
                        ctr = totals[key]

                        ctr['total'] += int(mask.sum())
                        ctr['usable'] += int(
                            (mask & valid).sum()
                        )
                        ctr['sentinel'] += int(
                            (mask & sentinel).sum()
                        )
                        ctr['out_of_range'] += int(
                            (mask & bad).sum()
                        )
                        ctr['blank'] += int(
                            (
                                mask &
                                ~(valid | sentinel | bad)
                            ).sum()
                        )

                        if arr is not None:
                            vals = arr[mask & valid]
                            if len(vals):
                                mins[key] = min(
                                    mins.get(key, np.inf),
                                    float(vals.min())
                                )
                                maxs[key] = max(
                                    maxs.get(key, -np.inf),
                                    float(vals.max())
                                )

                for group in np.unique(groups):
                    cohorts[
                        (str(group), 'raw_active_rows')
                    ] += int(
                        (
                            active &
                            (groups == group)
                        ).sum()
                    )

    for (feature, group), ctr in totals.items():
        records.append(dict(
            source='raw_performance_active_rows',
            vintage=year,
            period=group,
            feature=feature,
            total=ctr['total'],
            usable=ctr['usable'],
            blank=ctr['blank'],
            sentinel=ctr['sentinel'],
            out_of_range=ctr['out_of_range'],
            nonempty=ctr['usable'] + ctr['sentinel'],
            distinct=None,
            min_value=mins.get((feature, group)),
            max_value=maxs.get((feature, group))
        ))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    records, checks = [], Counter()
    cohorts = Counter()

    for year in YEARS:
        read_orig(year, records, checks)
        monthly_audit(year, records, checks, cohorts)
        read_perf_extras(year, records, checks, cohorts)

        print(
            f'{year}: '
            f'{checks[(year, "orig_rows")]:,} loans | '
            f'{checks[(year, "monthly_rows")]:,} monthly rows | '
            f'{checks[(year, "valid_labels")]:,} valid labels',
            flush=True
        )

    quality = pd.DataFrame(records)
    quality['usable_pct'] = (
        100 * quality['usable'] /
        quality['total'].replace(0, np.nan)
    ).round(2)

    quality.to_csv(
        OUT / '27c_feature_quality_by_vintage.csv',
        index=False
    )

    totals = quality.groupby(
        ['source', 'feature', 'period'],
        as_index=False
    )[
        ['total', 'usable', 'blank', 'sentinel', 'out_of_range']
    ].sum()

    totals['usable_pct'] = (
        100 * totals['usable'] /
        totals['total'].replace(0, np.nan)
    ).round(2)

    totals.to_csv(
        OUT / '27c_feature_quality_overview.csv',
        index=False
    )

    pd.DataFrame([
        {'vintage': y, 'check': k, 'value': v}
        for (y, k), v in sorted(checks.items())
    ]).to_csv(
        OUT / '27c_integrity_checks.csv',
        index=False
    )

    pd.DataFrame([
        {'period': p, 'metric': m, 'value': v}
        for (p, m), v in sorted(cohorts.items())
    ]).to_csv(
        OUT / '27c_period_coverage.csv',
        index=False
    )

    total_orig = sum(
        checks[(y, 'orig_rows')] for y in YEARS
    )
    total_base = sum(
        checks[(y, 'monthly_rows')] for y in YEARS
    )
    total_perf = sum(
        checks[(y, 'performance_rows')] for y in YEARS
    )
    total_labels = sum(
        checks[(y, 'valid_labels')] for y in YEARS
    )

    failures = {
        k: sum(checks[(y, k)] for y in YEARS)
        for k in [
            'history_90_flag_inconsistent',
            'history_count_exceeds_observations',
            'eligible_terminal_origin',
            'eligible_unknown_destination'
        ]
    }

    lines = [
        f'Origination loans: {total_orig:,} (expected 850,000)',
        f'Monthly base rows: {total_base:,} (expected 51,743,369)',
        f'Raw performance rows: {total_perf:,} (expected 51,743,369)',
        f'Valid next-month labels: {total_labels:,} (expected 50,892,470)',
        *[f'{k}: {v:,}' for k, v in failures.items()],
        '',
        'LIMITATIONS / CONTROLS:',
        'Origination coverage is loan-weighted; monthly coverage is active-row-weighted.',
        'Blank flags can mean no event/not applicable; do not equate nonblank rate with data completeness.',
        'Estimated LTV only disclosed since 2017-04; historical as-of AVM availability not proven.',
        'Performance data begins at Freddie Mac acquisition, not necessarily at loan origination.',
        'Historic data files may be revised; as-of publication availability not independently established.',
        'Monthly state at period t may reflect reporting lag; define model deployment as-of date.',
        'Borrower assistance/modification status are not known in future horizon without policy assumptions.',
        'Latest servicer name, MI cancellation, actual loss, final exit fields must not be PD predictors.',
        'Unemployment, HPI and benchmark mortgage rates are not yet merged: NOT AUDITED.',
        'Amounts on terminal rows are intentionally excluded from active-loan coverage diagnostics.'
    ]

    (
        OUT / '27c_audit_readme.txt'
    ).write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8'
    )

    print('\n=== FINAL CHECKS ===')

    for title, actual, expected in [
        ('Origination loans', total_orig, 850000),
        ('Monthly base rows', total_base, 51743369),
        ('Raw performance rows', total_perf, 51743369),
        ('Valid labels', total_labels, 50892470)
    ]:
        print(
            f'{title}: {actual:,} | '
            f'{"PASS" if actual == expected else "CHECK"}'
        )

    for k, v in failures.items():
        print(
            f'{k}: {v:,} | '
            f'{"PASS" if v == 0 else "REVIEW"}'
        )

    print('\n=== SELECTED FEATURE COVERAGE (%) ===')

    selected = [
        'credit_score',
        'original_ltv',
        'original_dti',
        'amortization_type',
        'upb_t',
        'interest_rate_pct_t',
        'estimated_ltv',
        'payment_deferral_flag',
        'borrower_assistance'
    ]

    subset = totals[
        totals.feature.isin(selected)
    ].copy()

    print(
        subset[
            ['source', 'feature', 'period', 'usable_pct']
        ].to_string(index=False)
    )

    print(
        '\nSaved: 27c_feature_quality_by_vintage.csv, '
        '27c_feature_quality_overview.csv, '
        '27c_integrity_checks.csv, '
        '27c_period_coverage.csv, '
        '27c_audit_readme.txt'
    )


if __name__ == '__main__':
    main()
