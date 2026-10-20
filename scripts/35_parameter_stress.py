
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT / "outputs/modeling/task33_loss"
    / "task34_loan_inputs.csv"
)

OUT = ROOT / "outputs/modeling/task35_stress"

# Audited historical reference parameters
Q_REF = 10396 / 24324
LGD_REF = 0.4645

# Purely illustrative parameter shocks
# (PD multiplier, q multiplier, LGD multiplier)
SCENARIOS = {
    "Reference": (1.00, 1.00, 1.00),
    "PD +25%": (1.25, 1.00, 1.00),
    "PD +50%": (1.50, 1.00, 1.00),
    "Conversion +25%": (1.00, 1.25, 1.00),
    "LGD +20%": (1.00, 1.00, 1.20),
    "Combined Stress": (1.50, 1.25, 1.20),
}


def main():

    print("=== TASK 35: PARAMETER STRESS SENSITIVITY ===")

    if not SRC.exists():
        raise FileNotFoundError(SRC)

    df = pd.read_csv(SRC)

    if len(df) != 30000:
        raise ValueError("Unexpected number of loans")

    pd12 = pd.to_numeric(
        df["pd_12m"], errors="raise"
    ).to_numpy(dtype=float)

    upb = pd.to_numeric(
        df["upb_asof"], errors="raise"
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(pd12).all()
        or not np.isfinite(upb).all()
        or np.any(pd12 < 0)
        or np.any(pd12 > 1)
        or np.any(upb < 0)
    ):
        raise ValueError("Invalid input data")

    if not np.isclose(
        np.dot(pd12, upb),
        28_418_097.39,
        rtol=0.001
    ):
        raise ValueError("Task 33 exposure mismatch")

    total_upb = float(upb.sum())

    summary_rows = []
    state_rows = []

    for name, (pd_mult, q_mult, lgd_mult) in SCENARIOS.items():

        # Stress the marginal First90 probability
        pd_stressed = np.minimum(pd12 * pd_mult, 1.0)

        q = min(Q_REF * q_mult, 1.0)
        lgd = LGD_REF * lgd_mult

        if lgd > 1:
            print(
                "WARNING: LGD exceeds 100%:",
                name
            )

        exposure = pd_stressed * upb
        loan_loss = exposure * q * lgd

        loss = float(loan_loss.sum())

        summary_rows.append({
            "scenario": name,
            "pd_multiplier": pd_mult,
            "q_multiplier": q_mult,
            "lgd_multiplier": lgd_mult,
            "stressed_q": q,
            "stressed_lgd": lgd,
            "expected_first90_count":
                float(pd_stressed.sum()),
            "first90_exposure_usd":
                float(exposure.sum()),
            "loss_proxy_usd": loss,
            "loss_pct_upb": loss / total_upb * 100,
        })

        states = df["state_t"].astype(str)

        temp = pd.DataFrame({
            "state_t": states,
            "loss_proxy_usd": loan_loss,
            "exposure_usd": exposure,
        })

        grouped = temp.groupby(
            "state_t", as_index=False
        ).agg(
            exposure_usd=("exposure_usd", "sum"),
            loss_proxy_usd=("loss_proxy_usd", "sum"),
        )

        grouped.insert(0, "scenario", name)

        state_rows.append(grouped)

    summary = pd.DataFrame(summary_rows)
    by_state = pd.concat(state_rows, ignore_index=True)

    reference = float(
        summary.loc[
            summary["scenario"] == "Reference",
            "loss_proxy_usd"
        ].iloc[0]
    )

    if not np.isclose(
        reference, 5_641_726,
        rtol=0.001
    ):
        raise ValueError("Task 33.2 reference mismatch")

    summary["loss_increase_vs_reference_pct"] = (
        (summary["loss_proxy_usd"] / reference - 1)
        * 100
    )

    # Reconcile subgroup totals
    for _, row in summary.iterrows():
        subtotal = by_state.loc[
            by_state.scenario == row.scenario,
            "loss_proxy_usd"
        ].sum()

        if not np.isclose(
            subtotal, row.loss_proxy_usd, atol=0.01
        ):
            raise ValueError("State reconciliation failed")

    OUT.mkdir(parents=True, exist_ok=True)

    summary_path = OUT / "task35_stress_summary.csv"
    state_path = OUT / "task35_stress_by_state.csv"
    chart_path = OUT / "task35_stress_chart.png"
    assumptions_path = OUT / "task35_assumptions.json"

    summary.to_csv(summary_path, index=False)
    by_state.to_csv(state_path, index=False)

    assumptions = {
        "snapshot": "2019-01",
        "portfolio_loans": 30000,
        "risk_event": "First90 within 12M",
        "historical_q_60m": Q_REF,
        "historical_lgd_reference": LGD_REF,
        "scenarios": SCENARIOS,
        "classification": "Parameter sensitivity only",
        "important_limitations": [
            "Not an empirically calibrated macroeconomic stress test",
            "Shocks are illustrative, not derived from a historical crisis",
            "PD changes are mechanical, not model re-predictions",
            "PD is capped at 100 percent",
            "The 2019 development PD overpredicts actual First90",
            "Historical q and LGD cohorts are not fully reconciled",
            "Historical parameters may include post-2019 information",
            "UPB is an as-of exposure proxy",
            "Potential eventual loss, not 12M booked loss",
            "Does not model joint credit dependence or economic capital"
        ]
    }

    assumptions_path.write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8"
    )

    # Chart for final GitHub report
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        summary["scenario"],
        summary["loss_proxy_usd"] / 1e6
    )

    ax.invert_yaxis()
    ax.set_xlabel("Potential eventual loss proxy (USD millions)")
    ax.set_title(
        "Mortgage Portfolio: Parameter Stress Sensitivity"
    )

    for i, val in enumerate(summary["loss_proxy_usd"]):
        ax.text(
            val / 1e6 + 0.08,
            i,
            f"${val / 1e6:.2f}M",
            va="center"
        )

    ax.set_xlim(
        0,
        summary["loss_proxy_usd"].max() / 1e6 * 1.22
    )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    print("\n=== TASK 35 RESULTS ===")

    display_cols = [
        "scenario",
        "expected_first90_count",
        "loss_proxy_usd",
        "loss_pct_upb",
        "loss_increase_vs_reference_pct"
    ]

    print(
        summary[display_cols].round(4).to_string(index=False)
    )

    print("\n=== OUTPUT FILES ===")
    for path in [
        summary_path,
        state_path,
        chart_path,
        assumptions_path
    ]:
        print(path)

    drive = Path("/content/drive/MyDrive")

    if drive.exists():

        backup = drive / "creditport-task35-backup"
        backup.mkdir(parents=True, exist_ok=True)

        for path in [
            summary_path,
            state_path,
            chart_path,
            assumptions_path,
            Path(__file__)
        ]:
            shutil.copy2(path, backup / path.name)

        print("\nGoogle Drive backup:", backup)

    else:
        print("\nWARNING: Google Drive not mounted")


if __name__ == "__main__":
    main()
