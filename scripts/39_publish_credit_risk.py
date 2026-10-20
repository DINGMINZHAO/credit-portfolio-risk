#!/usr/bin/env python3
"""Task 39: assemble independently tested OOT reporting and curated credit-risk visuals.

Run on the original Mac development project; no model training or prediction.
Requires a zip exported from Task 38 in Colab and existing historical audit outputs.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = 'tab:blue'
ORANGE = 'tab:orange'
RED = 'tab:red'
DARK = '#272727'
GRAY = '#666666'

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelcolor': DARK,
    'text.color': DARK,
    'axes.titleweight': 'bold',
})

PACKAGE_FILES = {
    'task38_oot_validation.md': 'reports/task38_oot_validation.md',
    'task38_oot_summary.csv': 'reports/tables/task38_oot_summary.csv',
    'task38_oot_breakdown.csv': 'reports/tables/task38_oot_breakdown.csv',
    'task38_development_vs_oot.csv': 'reports/tables/task38_development_vs_oot.csv',
    'task38_frozen_protocol.json': 'reports/validation/task38_frozen_protocol.json',
    'task38_data_audit.json': 'reports/validation/task38_data_audit.json',
    'task38_first90_calibration.png': 'reports/figures/task38_first90_calibration.png',
    'task38_oot_event_calibration.png': 'reports/figures/task38_oot_event_calibration.png',
}


def save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  figure:', path.name)


def plot_vintage(root: Path, figdir: Path):
    csv = root / 'outputs/07_horizon_summary.csv'
    dest = figdir / 'historical_first90_by_vintage.png'
    if csv.is_file():
        df = pd.read_csv(csv)
        needed = {'vintage','horizon','observed_first90_pct','eligible','incomplete'}
        if not needed.issubset(df.columns):
            raise ValueError(f'{csv} missing columns: {sorted(needed-set(df.columns))}')
        fig, ax = plt.subplots(figsize=(11.5,5.8))
        for horizon,color in [(12,BLUE),(24,ORANGE),(36,RED)]:
            s = df.loc[df.horizon.eq(horizon)].sort_values('vintage')
            ax.plot(s.vintage, s.observed_first90_pct, marker='o', markersize=4,
                    linewidth=2.2, label=f'{horizon}-month',color=color)
        ax.set(title='Observed First90 incidence by origination vintage',
               xlabel='Origination year',ylabel='Observed First90 share of eligible loans (%)')
        ax.set_xticks(sorted(df.vintage.dropna().astype(int).unique()))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y',alpha=.20)
        ax.legend(frameon=False,ncol=3)
        ax.text(0,-0.20,'Fixed-horizon observed shares; incomplete follow-up is tracked separately (not a survival estimate).',
                transform=ax.transAxes,fontsize=9,color=GRAY)
        fig.tight_layout()
        save(fig,dest)
    else:
        fallback=root/'outputs/figures/07_first90_by_vintage.png'
        if not fallback.is_file(): raise FileNotFoundError(f'Neither {csv} nor {fallback} exists')
        shutil.copy2(fallback,dest)
        print('  reused original:',dest.name)


def plot_calendar(root: Path, figdir: Path):
    csv = root / 'outputs/08_vintage_year_first90.csv'
    dest=figdir/'historical_vintage_calendar_heatmap.png'
    if csv.is_file():
        df=pd.read_csv(csv)
        required={'vintage','year','risk_loan_months','per_loan_month_rate_pct'}
        if not required.issubset(df.columns):
            raise ValueError(f'{csv} missing columns: {sorted(required-set(df.columns))}')
        df=df.loc[df['risk_loan_months'].ge(1000)]
        years=np.arange(int(df.year.min()),int(df.year.max())+1)
        vintages=np.arange(int(df.vintage.min()),int(df.vintage.max())+1)
        matrix=np.full((len(vintages),len(years)),np.nan)
        for row in df.itertuples(index=False):
            matrix[int(row.vintage)-vintages[0],int(row.year)-years[0]]=float(row.per_loan_month_rate_pct)
        fig,ax=plt.subplots(figsize=(12.5,6.2))
        im=ax.imshow(np.ma.masked_invalid(matrix),origin='lower',aspect='auto',cmap='OrRd')
        ix=np.arange(0,len(years),2)
        ax.set_xticks(ix,years[ix])
        ax.set_yticks(np.arange(len(vintages)),vintages)
        ax.set(title='Monthly First90 incidence | origination vintage × calendar year',
               xlabel='Calendar year',ylabel='Origination vintage')
        cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.02)
        cb.set_label('Events per 100 at-risk loan-months')
        ax.text(0,-0.17,'Cells with fewer than 1,000 at-risk loan-months are suppressed.',
                transform=ax.transAxes,fontsize=9,color=GRAY)
        fig.tight_layout()
        save(fig,dest)
    else:
        fallback=root/'outputs/figures/08_vintage_calendar_heatmap.png'
        if not fallback.is_file(): raise FileNotFoundError(f'Neither {csv} nor {fallback} exists')
        shutil.copy2(fallback,dest)
        print('  reused original:',dest.name)


def plot_lgd(root: Path, figdir: Path):
    parquet=root/'data/processed/20_unified_lgd_2006_2022.parquet'
    dest=figdir/'historical_lgd_distribution.png'
    if parquet.is_file():
        df=pd.read_parquet(parquet,columns=['exit_code','valid_lgd','lgd_signed','actual_loss','upb_at_first90'])
        code=(df.exit_code.astype('string').str.strip().str.replace(r'\.0$', '',regex=True).str.zfill(2))
        selected=df.loc[code.isin(['02','03','09']) & df.valid_lgd.fillna(False)].copy()
        lgd=pd.to_numeric(selected.lgd_signed,errors='coerce').dropna()
        lgd=lgd[np.isfinite(lgd)]
        if len(lgd)<100: raise ValueError('Valid historical LGD sample unexpectedly small')
        upb=pd.to_numeric(selected.upb_at_first90,errors='coerce').sum()
        loss=pd.to_numeric(selected.actual_loss,errors='coerce').sum()
        weighted=loss/upb
        visible=lgd[lgd.between(-.5,1.5)]
        fig,ax=plt.subplots(figsize=(11.5,5.8))
        ax.hist(visible*100,bins=np.linspace(-50,150,75),color=BLUE,alpha=.84)
        ax.axvline(lgd.median()*100,color=ORANGE,linewidth=2,linestyle='--',label=f'Median {lgd.median():.1%}')
        ax.axvline(weighted*100,color=RED,linewidth=2,linestyle='--',label=f'UPB-weighted reference {weighted:.1%}')
        ax.set(title='Historical loss-given-default distribution | credit resolutions 02 / 03 / 09',
               xlabel='Signed LGD (%)',ylabel='Resolved loans')
        ax.grid(axis='y',alpha=.17)
        ax.legend(frameon=False)
        ax.text(.98,.92, f'Valid loans: {len(lgd):,}\nNegative LGD: {(lgd<0).mean():.1%}\nAbove 100%: {(lgd>1).mean():.1%}',
                transform=ax.transAxes,ha='right',va='top',fontsize=10,
                bbox=dict(boxstyle='round,pad=.5',facecolor='white',edgecolor='#DDDDDD'))
        ax.text(0,-.20,'Displayed range: -50% to 150%; tail observations outside range are excluded from bars only.',
                transform=ax.transAxes,fontsize=9,color=GRAY)
        fig.tight_layout()
        save(fig,dest)
    else:
        fallback=root/'outputs/lgd_audit/21_lgd_overall_hist.png'
        if not fallback.is_file(): raise FileNotFoundError(f'Neither {parquet} nor {fallback} exists')
        shutil.copy2(fallback,dest)
        print('  reused original:',dest.name)


def replace_section(txt: str, old_heading: str, replacement: str)->str:
    pattern=rf'(?ms)^{re.escape(old_heading)}\n.*?(?=^## |\Z)'
    new,num=re.subn(pattern,lambda m:replacement.rstrip()+'\n\n',txt)
    if num!=1:
        raise ValueError(f'Expected exactly one section headed {old_heading!r}; found {num}')
    return new


def extract_single(csv: Path,model: str,group: str):
    df=pd.read_csv(csv)
    q=df.loc[(df.model==model)&(df.group==group)]
    if len(q)!=1:raise ValueError(f'{csv.name}: unexpected match for {model}/{group}')
    return q.iloc[0]


def render_readme(readme: str,table:Path)->str:
    dev_no=extract_single(table,'nomacro','all')
    dev_yes=extract_single(table,'withmacro','all')
    no21=extract_single(table,'nomacro','year_2021')
    no22=extract_single(table,'nomacro','year_2022')
    yes21=extract_single(table,'withmacro','year_2021')
    yes22=extract_single(table,'withmacro','year_2022')
    if int(dev_no['rows'])!=6140535 or int(dev_yes['rows'])!=6140535:
        raise ValueError('Unexpected OOT row count')
    if int(dev_no['first90_actual'])!=3034 or int(dev_yes['first90_actual'])!=3034:
        raise ValueError('Unexpected OOT First90 count')

    historic='''## Historical mortgage credit behavior

The historical audits trace how first severe delinquency varies across origination cohorts and calendar periods. All figures use the original Freddie Mac sample and audited event definitions.

![Observed First90 share by origination vintage and fixed observation horizon](reports/figures/historical_first90_by_vintage.png)

**Origination-vintage analysis.** The 12-, 24-, and 36-month observed First90 shares reveal how severe-delinquency incidence differs by loan cohort and follow-up horizon. Incomplete observation is accounted for separately in the underlying audit.

![First90 event rates by vintage and calendar year](reports/figures/historical_vintage_calendar_heatmap.png)

**Calendar-time analysis.** Monthly First90 event rates per at-risk loan-month show how credit outcomes vary across both cohort and calendar period. Low-exposure cells are suppressed for reliability.
'''

    workflow='''## Modeling approach

The project combines audited loan histories, eight-state monthly transition modeling, 12-month First90 path simulation, historical severity analysis, and portfolio loss scenarios. State-specific LightGBM models predict movement among S0 Current, S1 30–59 DPD, S2 60–89 DPD, S3 90+ DPD, S4 Foreclosure, S5 Payoff, S6 Credit Resolution, and S7 Other Exit.

[View end-to-end workflow](reports/figures/project_workflow.png) · [View credit-state framework](reports/figures/credit_state_framework.png)
'''

    model=f'''## Model development and independent OOT validation

The modeling pipeline compares a multinomial logistic baseline with state-specific LightGBM variants, with and without lag-aware macroeconomic predictors.

![Development-period model comparison](reports/figures/model_comparison.png)

| Development metric (2018–2020) | Multinomial M3 | No Macro | With Macro |
|---|---:|---:|---:|
| Multiclass Logloss ↓ | 0.12760 | 0.12450 | 0.12145 |
| Multiclass Brier ↓ | 0.05007 | 0.04938 | 0.04901 |
| First90 O/E (ideal = 1) | 1.15716 | 1.06243 | 1.08942 |

**Independent 2021–2022 OOT evaluation.** Ten frozen LightGBM models were evaluated on **6,140,535 monthly transitions** spanning 24 calendar months. Neither training nor probability recalibration was repeated.

| OOT metric | No Macro | With Macro |
|---|---:|---:|
| Multiclass Logloss ↓ | {dev_no['logloss']:.5f} | {dev_yes['logloss']:.5f} |
| Multiclass Brier ↓ | {dev_no['brier']:.5f} | {dev_yes['brier']:.5f} |
| First90 actual / expected | {int(dev_no['first90_actual']):,} / {dev_no['first90_expected']:,.1f} | {int(dev_yes['first90_actual']):,} / {dev_yes['first90_expected']:,.1f} |
| First90 O/E (ideal = 1) | **{dev_no['first90_OE']:.4f}** | {dev_yes['first90_OE']:.4f} |

![Annual First90 calibration across development and independent OOT](reports/figures/task38_first90_calibration.png)

**Key finding:** The macro-enhanced LightGBM achieved lower overall multiclass prediction error, while the no-macro variant delivered more accurate First90 event-count calibration in the independent holdout. Annual First90 O/E was **{no21['first90_OE']:.3f} / {no22['first90_OE']:.3f}** for No Macro versus **{yes21['first90_OE']:.3f} / {yes22['first90_OE']:.3f}** for With Macro (2021 / 2022).

![Observed versus expected OOT First90 and exit-state event counts](reports/figures/task38_oot_event_calibration.png)

**State-level validation:** Additional observed-versus-expected tests identified concentrated calibration gaps in credit resolution (S6), particularly from severe delinquency (S3). This shows why business-relevant events must be evaluated in addition to aggregate predictive scores.

**Validation context:** The development results include 2018 observations used for early stopping and run through October 2020. The independent OOT covers 2021–2022. The separate January 2019 **12-month** First90 path simulation estimated 155.52 events against 111 observed; that multi-period simulation was not independently OOT-tested.

[Read independent OOT validation report](reports/task38_oot_validation.md) · [Review frozen-model OOT summary](reports/tables/task38_oot_summary.csv) · [Review models](models/)
'''

    lgd='''## Historical loss severity

![Historical signed loss-given-default distribution](reports/figures/historical_lgd_distribution.png)

Audited credit-resolution records show the distribution of realized loss severity, including recoveries and loss amounts above the first-90+ loan balance. The exposure-weighted historical LGD reference is **46.45%**; its sample definition is documented in the [full results report](reports/credit_portfolio_risk_results.md).
'''

    scope='''## Model scope and interpretation

- **Monthly-model evidence:** Both LightGBM variants have been independently tested on 2021–2022 monthly transitions. The 12-month simulated First90 scenario remains a separately assessed development-period result.
- **Portfolio loss interpretation:** Exposure, historical resolution conversion and LGD are combined as transparent loss proxies; the reported tail-risk estimates use assumed dependence rather than calibrated mortgage default correlation.
- **Data context:** Historical macro variables use conservative publication-lag proxies rather than archived real-time vintages. Equal-size origination-vintage sampling is an analytical sample, not the naturally weighted US mortgage market.

For methodology, evaluation details and usage boundaries, see [Credit Portfolio Risk Results](reports/credit_portfolio_risk_results.md) and [Independent OOT Validation](reports/task38_oot_validation.md).
'''

    readme=readme.replace(
        "> **Project status:** Research / analytics prototype for a sampled mortgage portfolio. The loss numbers below are **illustrative conditional proxies**, not validated IFRS 9 / CECL expected credit losses, regulatory capital, or production forecasts.",
        "> **Project scope:** End-to-end mortgage credit analytics and independent monthly model validation, with transparent portfolio-loss scenario estimates."
    )
    readme=readme.replace(
        "**Business question:** How can a lender translate granular mortgage delinquency dynamics into interpretable portfolio risk estimates—and identify the assumptions most likely to break those estimates?",
        "**Business question:** How can a lender turn loan-level delinquency histories into actionable credit-risk insights, evaluate their stability across economic conditions, and connect them to portfolio loss scenarios?\n\n**Independent validation highlight:** On 6.14 million 2021–2022 monthly transitions, No Macro achieved **First90 O/E = 1.011**, while With Macro achieved the lower multiclass log loss (**0.12201**)."
    )
    if '## Historical mortgage credit behavior' in readme:
        raise ValueError('README already has Task 39 additions. Do not apply twice.')
    readme=readme.replace('## End-to-end analytical workflow',historic+'\n## End-to-end analytical workflow',1)
    # Combine the two existing conceptual sections into a short approach overview.
    span=re.compile(r'(?ms)^## End-to-end analytical workflow\n.*?(?=^## Model development and validation\n)')
    readme,n=span.subn(lambda m:workflow+'\n',readme)
    if n!=1:raise ValueError('Unable to find original workflow + state sections')
    readme=replace_section(readme,'## Model development and validation',model)
    readme=readme.replace('## Connecting PD to portfolio exposure and loss',lgd+'\n## Connecting PD to portfolio exposure and loss',1)
    readme=replace_section(readme,'## Model risk, assumptions, and limitations',scope)
    old='**Figures and reporting:** The four additional README figures are generated from the released summary tables and model validation metrics, with no raw mortgage records needed:'
    new='**Figures and reporting:** Summary cards and the development comparison can be regenerated from published result tables. Historical audit plots and OOT figures are provided as fixed, auditable outputs alongside the corresponding scripts and reports:'
    if old not in readme:raise ValueError('Original reproducibility sentence not found')
    readme=readme.replace(old,new)
    orig='| [`reports/`](reports) | Detailed written report, charts, and summary tables |'
    rep='| [`reports/`](reports) | Detailed report, historical visualizations, portfolio scenarios, and independent [OOT validation](reports/task38_oot_validation.md) |'
    if orig not in readme:raise ValueError('Original navigation row not found')
    readme=readme.replace(orig,rep)
    return readme


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.home()/'Desktop/credit-portfolio-data')
    ap.add_argument('--oot-zip',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.expanduser().resolve()
    zip_path=args.oot_zip.expanduser().resolve()
    readme=root/'README.md'
    if not readme.is_file():raise FileNotFoundError(f'No README: {readme}')
    if not zip_path.is_file():raise FileNotFoundError(f'No OOT package: {zip_path}')
    if not (root/'models/nomacro/manifest.json').exists():raise FileNotFoundError('Run inside the original portfolio project')

    with ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad is not None:raise ValueError(f'Corrupted package member: {bad}')
        missing=set(PACKAGE_FILES)-set(z.namelist())
        if missing:raise ValueError(f'OOT package missing: {sorted(missing)}')
        # Extract into memory first, then validate OOT totals before mutating repository.
        bytes_by_name={name:z.read(name) for name in PACKAGE_FILES}

    summary=pd.read_csv(io.BytesIO(bytes_by_name['task38_oot_summary.csv']))
    for variant in ('nomacro','withmacro'):
        row=summary.loc[(summary.model==variant)&(summary.group=='all')]
        if len(row)!=1 or int(row.iloc[0]['rows'])!=6140535 or int(row.iloc[0]['first90_actual'])!=3034:
            raise ValueError(f'OOT summary verification failed for {variant}')
    report=bytes_by_name['task38_oot_validation.md'].decode('utf-8')
    if '2021' not in report or '2022' not in report:
        raise ValueError('Unexpected OOT report contents')

    original=readme.read_text(encoding='utf-8')
    # Render using exact OOT breakdown, staged in a temporary location only.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        staged=Path(temp)/'oot.csv'
        staged.write_bytes(bytes_by_name['task38_oot_breakdown.csv'])
        rewritten=render_readme(original,staged)

    figures=root/'reports/figures'
    figures.mkdir(parents=True,exist_ok=True)
    print('Generating historical data visualizations...')
    # Fail fast if historical sources are absent; no guessing.
    for f in (plot_vintage,plot_calendar,plot_lgd): f(root,figures)

    print('Installing audited OOT reports and figures...')
    for source,destination in PACKAGE_FILES.items():
        target=root/destination
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(bytes_by_name[source])
        print('  installed:',destination)

    backup=root/'outputs/task39_readme_before_release.md'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_text(original,encoding='utf-8')
    readme.write_text(rewritten,encoding='utf-8')
    print('README updated; original backed up at:',backup)
    # Retain the figure-generation and publishing source in the repository.
    script_dest = root/'scripts/39_publish_credit_risk.py'
    script_dest.parent.mkdir(parents=True, exist_ok=True)
    if not script_dest.exists():
        shutil.copy2(Path(__file__).resolve(),script_dest)
    elif script_dest.read_bytes() != Path(__file__).resolve().read_bytes():
        print('NOTICE: existing scripts/39_publish_credit_risk.py differs; kept unchanged.')

    details=root/'reports/credit_portfolio_risk_results.md'
    if details.is_file():
        text=details.read_text(encoding='utf-8')
        obsolete='3. The reserved 2021-2023 OOT evaluation has not been reported.'
        if obsolete in text:
            text=text.replace(obsolete,'3. Independent monthly OOT validation for 2021–2022 is reported separately in [Task 38](task38_oot_validation.md); the 12-month simulator remains development-assessed.')
            details.write_text(text,encoding='utf-8')
            print('Updated outdated OOT statement in detailed results report.')

    # Validate local relative markdown figure paths.
    links=re.findall(r'!\[[^\]]*\]\((reports/figures/[^)]+)\)',rewritten)
    missing=[s for s in links if not (root/s).is_file()]
    if missing:raise RuntimeError(f'Unresolved image references: {missing}')
    print('\nTASK 39: STAGED — new data visuals, OOT assets and README ready.')
    print('Check: git diff --check && git diff --stat && git status --short')
    print('Review README in VS Code before committing. Nothing was pushed.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr)
        raise
