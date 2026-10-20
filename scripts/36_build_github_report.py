
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOSS = ROOT / "outputs/modeling/task33_loss"
SIM = ROOT / "outputs/modeling/task34_portfolio_loss"
STRESS = ROOT / "outputs/modeling/task35_stress"

REPORT = ROOT / "reports"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"

for folder in [REPORT, FIGURES, TABLES]:
    folder.mkdir(parents=True, exist_ok=True)

sources = {
    "loss": LOSS / "task33_2_loss_scenarios.csv",
    "simulation": SIM / "task34_summary.csv",
    "stress": STRESS / "task35_stress_summary.csv",
}

for name, path in sources.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")

loss = pd.read_csv(sources["loss"])
simulation = pd.read_csv(sources["simulation"])
stress = pd.read_csv(sources["stress"])

def one(df, column, value):
    rows = df.loc[df[column] == value]
    if len(rows) != 1:
        raise ValueError(f"Expected one {column}={value}, got {len(rows)}")
    return rows.iloc[0]

ref = one(loss, "scenario", "Historical Reference")
combined = one(stress, "scenario", "Combined Stress")
stress_ref = one(stress, "scenario", "Reference")

sim_ref = simulation.loc[
    simulation["scenario"] == "Historical Reference"
].copy()

independent = sim_ref.loc[
    np.isclose(sim_ref["latent_factor_rho_assumed"], 0.0)
]

correlated = sim_ref.loc[
    np.isclose(sim_ref["latent_factor_rho_assumed"], 0.15)
]

if len(independent) != 1 or len(correlated) != 1:
    raise ValueError("Required simulation scenarios not found")

independent = independent.iloc[0]
correlated = correlated.iloc[0]

# Verify consistency between the three completed tasks
reference_el = float(ref["loss_proxy_usd"])

checks = {
    "Task34 independent analytic EL":
        independent["analytic_expected_loss_usd"],
    "Task34 correlated analytic EL":
        correlated["analytic_expected_loss_usd"],
    "Task35 reference loss":
        stress_ref["loss_proxy_usd"],
}

for name, value in checks.items():
    if not np.isclose(value, reference_el, rtol=0.001):
        raise ValueError(f"Inconsistent {name}: {value}")

# Copy selected GitHub-ready results only.
# Do not copy raw Freddie Mac loan-level datasets.
for name, path in sources.items():
    shutil.copy2(path, TABLES / path.name)

figures = [
    SIM / "task34_loss_distribution.png",
    STRESS / "task35_stress_chart.png",
]

for path in figures:
    if not path.exists():
        raise FileNotFoundError(path)
    shutil.copy2(path, FIGURES / path.name)

def money(value):
    return f"${value / 1e6:,.2f}M"

def pct(value):
    return f"{value:.2f}%"

def sim_row(row):
    return (
        f"| {row['latent_factor_rho_assumed']:.2f} "
        f"| {money(row['simulated_mean_loss_usd'])} "
        f"| {money(row['unexpected_loss_std_usd'])} "
        f"| {money(row['VaR_99_usd'])} "
        f"| {money(row['ES_99_usd'])} |"
    )

scenario_table = "\n".join(
    f"| {r['scenario']} | {r['q_assumed']:.2%} "
    f"| {money(r['loss_proxy_usd'])} |"
    for _, r in loss.iterrows()
)

stress_table = "\n".join(
    f"| {r['scenario']} | {money(r['loss_proxy_usd'])} "
    f"| {r['loss_increase_vs_reference_pct']:+.2f}% |"
    for _, r in stress.iterrows()
)

report = f"""# Credit Portfolio Risk Modeling & Simulation

## Executive Summary

This project develops a loan-level credit risk framework using
Freddie Mac Single-Family Loan-Level Dataset observations.

The analytical pipeline combines monthly credit-state transition
modeling, 12-month cumulative first-90-day delinquency probability,
exposure proxies, historical loss references, Monte Carlo portfolio
loss simulation, and parameter stress sensitivity.

The reported results describe a **30,000-loan development sample
as of January 2019**, not a representative US mortgage market
portfolio or an independently validated production risk system.

## 1. Portfolio and Credit Risk

| Metric | Result |
|---|---:|
| Portfolio snapshot | January 2019 |
| Sample loans | 30,000 |
| Total as-of unpaid principal balance | {money(ref['total_upb_usd'])} |
| Expected first-90+ loans within 12 months | {ref['expected_first90_events']:.2f} |
| Observed first-90+ loans | 111 |
| 12M First90 observed/expected ratio | 0.7137 |
| Probability-weighted first-90+ exposure | {money(ref['first90_exposure_usd'])} |

The 12-month First90 predictions are based on a state-specific
LightGBM transition model with 64 simulated paths per loan.

The First90 event is a severe delinquency indicator and should not
automatically be equated with an ultimate realized credit loss.

The development-period 12M probability estimate overpredicts
observed severe delinquency events.

## 2. Historical Credit Loss References

Historical audits provide two reference quantities:

- 10,396 credit resolutions (codes 02/03/09) among 24,324
  eligible historical First90 loans, observed over 60 months.
- Historical exposure-weighted disposition-sample LGD of 46.45%.

The historical First90-to-resolution conversion is approximately
42.74%.

These historical quantities do not come from fully reconciled
identical cohorts. The historical LGD is not a validated prospective
LGD forecast, and the exposure uses January 2019 UPB rather
than default-time EAD.

### Loss Scenario Sensitivity

| Scenario | Conditional conversion assumption | Eventual loss proxy |
|---|---:|---:|
{scenario_table}

The historical reference loss proxy is **{money(reference_el)}**.

This is not a validated IFRS 9 ECL, CECL allowance, or measure of
losses recognized within the next 12 months.

## 3. Portfolio Loss Distribution

10,000 Monte Carlo scenarios are used to compare independent
loan-level events with a hypothetical one-factor Gaussian
dependence structure.

Credit-resolution conversion is modeled conditionally on First90,
and LGD is held fixed at the historical reference.

### Historical Reference Scenario

| Assumed latent-factor rho | Mean loss | Loss standard deviation | 99% VaR | 99% ES |
|---:|---:|---:|---:|---:|
{sim_row(independent)}
{sim_row(correlated)}

![Simulated loss distribution](figures/task34_loss_distribution.png)

Introducing a shared risk factor substantially widens the
simulated loss distribution, while the marginal PD-based
analytical expected loss remains unchanged.

The assumed latent-factor rho of 0.15 is not an estimated
mortgage default correlation. The resulting VaR and ES are
illustrative conditional scenario statistics, not validated
economic capital estimates.

## 4. Parameter Stress Sensitivity

The following shocks are illustrative mechanical adjustments
to PD, credit-resolution conversion, and LGD.

They are not produced by re-running the predictive model under
historical or forecast macroeconomic variables.

| Parameter scenario | Eventual loss proxy | Change versus reference |
|---|---:|---:|
{stress_table}

![Parameter sensitivity](figures/task35_stress_chart.png)

Under the combined illustrative parameter stress, the potential
eventual loss proxy increases to
**{money(combined['loss_proxy_usd'])}**, a
**{combined['loss_increase_vs_reference_pct']:.2f}%** increase
relative to the reference scenario.

## 5. Model Risk and Limitations

1. The First90 model overpredicts the observed development
   sample event count (O/E = 0.7137).
2. The 2018 data were used for LightGBM early stopping and
   cannot be presented as fully independent testing.
3. The reserved 2021-2023 OOT evaluation has not been reported.
4. The 12M path simulator fixes future macroeconomic inputs
   and uses simplified loan-state and balance dynamics.
5. As-of UPB is a proxy, not actual default-time EAD.
6. Historical conversion and LGD estimates are not verified
   on completely aligned cohorts.
7. Loan-level LGD variation and its dependence on the economy
   are omitted from the portfolio simulation.
8. The common-factor dependence parameter is hypothetical,
   not calibrated from historical observations.
9. Parameter stress scenarios are not validated macroeconomic
   stress tests.
10. Historical results based on the latest data release may
    contain information not available at the January 2019
    prediction date.
11. Each origination vintage was sampled at a fixed count,
    so aggregate results should not be generalized to the
    naturally weighted US mortgage market.
12. Simulated risk measures are not regulatory capital,
    accounting provisions, or independently validated forecasts.

## 6. Business Interpretation

The project demonstrates an end-to-end workflow connecting
loan-level credit-state probabilities to portfolio exposure,
conditional loss scenarios, and dependence-sensitive tail risk.

The central lesson is that a point estimate of expected loss
is insufficient for portfolio risk assessment: tail losses
depend materially on joint credit deterioration, loss severity,
and post-delinquency resolution assumptions.

The results are best interpreted as a transparent quantitative
risk analysis prototype with explicit model-risk limitations.

## 7. Reproducibility

The associated scripts and output artifacts should be retained
with their exact versions. The supplied report tables come from
Tasks 33.2, 34, and 35; upstream LightGBM and 12M simulation
results were generated in earlier development tasks.

Raw Freddie Mac data are not included in this report package.
Data access and any redistribution must follow applicable
dataset terms.
"""

destination = REPORT / "credit_portfolio_risk_results.md"
destination.write_text(report, encoding="utf-8")

# Back up the report package and this script
drive = Path("/content/drive/MyDrive")

if drive.exists():
    backup = drive / "creditport-final-report"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(REPORT, backup)

    shutil.copy2(
        Path(__file__),
        backup / Path(__file__).name
    )

    print("Drive backup:", backup)
else:
    print("WARNING: Google Drive not mounted")

print("\n=== FINAL REPORT QA ===")
print("Task 33-35 reference loss reconciled.")
print("Reference loss:", money(reference_el))
print("Combined stress:", money(combined["loss_proxy_usd"]))
print("Independent 99% VaR:", money(independent["VaR_99_usd"]))
print("Correlated 99% VaR:", money(correlated["VaR_99_usd"]))

print("\nReport generated:")
print(destination)
print("\nTask 36 completed.")
