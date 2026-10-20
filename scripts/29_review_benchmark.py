
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop/credit-portfolio-data"
OUT = ROOT / "outputs/modeling/task29_benchmark"


def ratio(a, b):
    return np.where(b != 0, a / b, np.nan)


def main():
    monthly = pd.read_csv(OUT / "validation_monthly.csv")
    states = pd.read_csv(OUT / "validation_by_next_state.csv")
    origins = pd.read_csv(OUT / "validation_by_start_state.csv")

    # 1. Year-by-year event calibration
    monthly["year"] = monthly["period_yyyymm"] // 100

    cols = [
        "first90_risk_rows",
        "first90_actual",
        "first90_expected",
        "credit_rows",
        "credit_actual",
        "credit_expected",
    ]

    annual = (
        monthly.groupby(["model", "year"], as_index=False)[cols]
        .sum()
    )

    annual["first90_OE"] = ratio(
        annual["first90_actual"],
        annual["first90_expected"],
    )

    annual["credit_exit_OE"] = ratio(
        annual["credit_actual"],
        annual["credit_expected"],
    )

    annual["first90_actual_rate_pct"] = (
        100 * annual["first90_actual"]
        / annual["first90_risk_rows"]
    )

    annual["first90_pred_rate_pct"] = (
        100 * annual["first90_expected"]
        / annual["first90_risk_rows"]
    )

    print("\n=== 1. ANNUAL CALIBRATION ===")

    print(
        annual[
            [
                "model", "year",
                "first90_actual_rate_pct",
                "first90_pred_rate_pct",
                "first90_OE",
                "credit_exit_OE",
            ]
        ].round(4).to_string(index=False)
    )

    # 2. Model B: calibration by next state
    print("\n=== 2. MODEL B: NEXT-STATE CALIBRATION ===")

    b = states[
        states["model"] == "state_plus_history"
    ].copy()

    print(
        b[
            ["next_state", "actual", "expected", "OE"]
        ].round(4).to_string(index=False)
    )

    # 3. Compare prediction quality by starting state
    comparison = origins.pivot(
        index="starting_state",
        columns="model",
        values="logloss",
    )

    comparison["improvement_pct"] = (
        100 * (
            comparison["state_only"]
            - comparison["state_plus_history"]
        ) / comparison["state_only"]
    )

    print("\n=== 3. LOG LOSS BY STARTING STATE ===")

    print(
        comparison.round(4).to_string()
    )

    # Save one consolidated report
    annual.to_csv(
        OUT / "annual_calibration_review.csv",
        index=False,
    )

    print("\nReview completed. No models retrained.")


if __name__ == "__main__":
    main()
