# Credit Portfolio Risk Modeling & Simulation

An end-to-end mortgage credit risk analytics project
using the Freddie Mac Single-Family Loan-Level Dataset.

## Project Overview

This project connects loan-level credit-state transition
modeling to 12-month severe delinquency probabilities,
portfolio exposure, conditional loss scenarios,
and portfolio tail-risk simulation.

## Dataset

- Freddie Mac Single-Family Loan-Level Dataset (Release 47)
- Origination vintages: 2006–2022
- Total sampled loans: 850,000
- Monthly performance records: approximately 51.7 million
- Portfolio snapshot: January 2019
- Portfolio analysis sample: 30,000 loans

The sampled dataset is not a naturally weighted
representation of the US mortgage market.

## Methodology

1. Monthly credit-state transition modeling with LightGBM.
2. Twelve-month cumulative First90 probability simulation.
3. Probability-weighted exposure using as-of UPB.
4. Historical credit-resolution and LGD analysis.
5. Monte Carlo portfolio loss simulation.
6. Parameter stress sensitivity.

## Key Results

| Metric | Result |
|---|---:|
| Sample loans | 30,000 |
| Portfolio UPB | $5.492B |
| Mean 12M First90 PD | 0.5184% |
| Predicted First90 events | 155.52 |
| Observed First90 events | 111 |
| First90 O/E | 0.7137 |
| Probability-weighted exposure proxy | $28.42M |
| Historical reference loss proxy | $5.64M |
| Independent simulation 99% VaR | $7.55M |
| Shared-factor scenario 99% VaR | $25.35M |
| Combined parameter stress loss proxy | $12.66M |

## Portfolio Loss Distribution

![Loss distribution](reports/figures/task34_loss_distribution.png)

## Parameter Stress Sensitivity

![Stress sensitivity](reports/figures/task35_stress_chart.png)

## Model Risk and Limitations

- First90 PD is not equivalent to realized credit loss.
- The 12M model overpredicts observed development events.
- Exposure uses as-of UPB rather than default-time EAD.
- Historical conversion and LGD cohorts are not fully aligned.
- Dependence parameters are assumed, not calibrated.
- Future macroeconomic inputs are held fixed in simulation.
- Stress scenarios are illustrative parameter sensitivities.
- Independent OOT validation has not been reported.
- Simulated VaR and ES are not validated economic capital.
- The results are not validated IFRS 9 or CECL estimates.

## Detailed Report

See [Credit Portfolio Risk Results]
(reports/credit_portfolio_risk_results.md)
 for detailed methodology, findings and limitations.

## Reproducibility

The workflow was developed in Google Colab.
Scripts require the original Freddie Mac data, intermediate
files, model artifacts, and the expected project environment.
The complete pipeline is not independently reproducible
from this lightweight publication package alone.

## Data Source

https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
