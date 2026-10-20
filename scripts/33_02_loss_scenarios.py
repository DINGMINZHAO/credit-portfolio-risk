
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs/modeling/task33_loss"
    / "12m_first90_exposure_201901_30k.csv"
)

OUT = ROOT / "outputs/modeling/task33_loss"

# Previously audited historical results
HIST_FIRST90 = 24324
HIST_CREDIT_RESOLUTION = 10396
HIST_LGD = 0.4645

# Sensitivity around historical conversion rate
SCENARIOS = {
    "Lower": 0.75,
    "Historical Reference": 1.00,
    "Higher": 1.25,
}


def get_pd(df):
    candidates = [
        "pd_12m",
        "pd_first90_12m",
        "first90_pd_12m",
    ]

    for name in candidates:
        if name in df.columns:
            values = pd.to_numeric(
                df[name], errors="raise"
            ).astype(float)
            return name, values

    raise ValueError(
        f"PD column not found. Columns: {df.columns.tolist()}"
    )


def get_upb(df):
    """Identify current UPB using audited Task33.1 total."""

    expected_total = 5_491_510_784.00

    candidates = [
        c for c in df.columns
        if "upb" in c.lower()
        and not any(
            word in c.lower()
            for word in [
                "orig", "first90", "expected",
                "weighted", "loss", "ratio"
            ]
        )
    ]

    matches = []

    for col in candidates:
        values = pd.to_numeric(
            df[col], errors="coerce"
        )

        if (
            values.notna().all()
            and np.isfinite(values).all()
            and (values >= 0).all()
            and abs(values.sum() - expected_total)
                / expected_total < 0.001
        ):
            matches.append((col, values.astype(float)))

    if not matches:
        raise ValueError(
            "Cannot identify as-of UPB. "
            f"Candidate columns: {candidates}. "
            "Please inspect Task33.1 CSV."
        )

    if len(matches) > 1:
        first = matches[0][1]
        for _, other in matches[1:]:
            if not np.allclose(first, other, atol=0.01):
                raise ValueError(
                    "Multiple nonidentical UPB columns "
                    "match the total. Stop rather than guess."
                )

    return matches[0]


def main():

    print("=== TASK 33.2: LOSS SCENARIOS ===")

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE, low_memory=False)

    if len(df) != 30000:
        raise ValueError(
            f"Expected 30,000 loans, found {len(df)}"
        )

    pd_col, pd12 = get_pd(df)
    upb_col, upb = get_upb(df)

    if (
        not np.isfinite(pd12).all()
        or not pd12.between(0, 1).all()
    ):
        raise ValueError("Invalid PD values")

    if "asof_yyyymm" in df.columns:
        periods = pd.to_numeric(df["asof_yyyymm"])
        if not periods.eq(201901).all():
            raise ValueError("Wrong snapshot date")

    if {"_vintage_file", "loan_id"}.issubset(df.columns):
        if df.duplicated(
            ["_vintage_file", "loan_id"]
        ).any():
            raise ValueError("Duplicate loan keys")

    # Reproduce Task 33.1 exposure
    exposure = pd12 * upb

    total_upb = float(upb.sum())
    expected_events = float(pd12.sum())
    total_exposure = float(exposure.sum())

    # Reconciliation with confirmed results
    if abs(expected_events - 155.52) > 0.3:
        raise ValueError("Task32 PD total mismatch")

    if (
        abs(total_exposure - 28_418_097.41)
        / 28_418_097.41 > 0.001
    ):
        raise ValueError("Task33.1 exposure mismatch")

    print(f"PD column: {pd_col}")
    print(f"UPB column: {upb_col}")
    print(f"Loans: {len(df):,}")
    print(f"Total UPB: ${total_upb:,.2f}")
    print(f"Expected First90: {expected_events:.2f}")
    print(f"First90 exposure: ${total_exposure:,.2f}")

    # Historical conversion rate
    q_hist = HIST_CREDIT_RESOLUTION / HIST_FIRST90

    print("\n=== HISTORICAL REFERENCE ===")
    print(f"First90 cohort: {HIST_FIRST90:,}")
    print(
        f"Credit resolutions within 60M: "
        f"{HIST_CREDIT_RESOLUTION:,}"
    )
    print(f"Historical conversion: {q_hist:.4%}")
    print(f"Historical disposition LGD: {HIST_LGD:.4%}")

    # Starting credit state
    if "state_t" in df.columns:
        states = (
            df["state_t"]
            .astype(str)
            .str.upper()
            .str.replace(r"\.0$", "", regex=True)
        )

        states = states.where(
            states.str.startswith("S"),
            "S" + states
        )
    else:
        states = pd.Series("UNKNOWN", index=df.index)

    # Loan-level data for Task 34
    loan_cols = [
        c for c in [
            "_vintage_file",
            "loan_id",
            "asof_yyyymm",
            "first90_12m_actual",
        ]
        if c in df.columns
    ]

    loans = df[loan_cols].copy().reset_index(drop=True)

    loans["state_t"] = states.to_numpy()
    loans["pd_12m"] = pd12.to_numpy()
    loans["upb_asof"] = upb.to_numpy()
    loans["first90_exposure_proxy"] = exposure.to_numpy()

    summary_rows = []
    state_rows = []

    for scenario, multiplier in SCENARIOS.items():

        q = min(1.0, q_hist * multiplier)

        loan_loss = (
            loans["first90_exposure_proxy"]
            * q
            * HIST_LGD
        )

        total_loss = float(loan_loss.sum())

        summary_rows.append({
            "scenario": scenario,
            "q_assumed": q,
            "lgd_reference": HIST_LGD,
            "total_upb_usd": total_upb,
            "expected_first90_events": expected_events,
            "first90_exposure_usd": total_exposure,
            "loss_proxy_usd": total_loss,
            "loss_pct_upb": (
                total_loss / total_upb * 100
            ),
        })

        temp = loans[[
            "state_t",
            "pd_12m",
            "upb_asof",
            "first90_exposure_proxy"
        ]].copy()

        temp["loss_proxy"] = loan_loss

        grouped = temp.groupby(
            "state_t", as_index=False
        ).agg(
            loans=("pd_12m", "size"),
            expected_events=("pd_12m", "sum"),
            upb=("upb_asof", "sum"),
            exposure=("first90_exposure_proxy", "sum"),
            loss_proxy=("loss_proxy", "sum"),
        )

        grouped.insert(0, "scenario", scenario)
        state_rows.append(grouped)

    summary = pd.DataFrame(summary_rows)
    by_state = pd.concat(state_rows, ignore_index=True)

    # Check portfolio vs state-level aggregation
    for _, row in summary.iterrows():
        state_total = by_state.loc[
            by_state["scenario"] == row["scenario"],
            "loss_proxy"
        ].sum()

        if not np.isclose(
            state_total,
            row["loss_proxy_usd"],
            atol=0.01
        ):
            raise ValueError("State aggregation mismatch")

    OUT.mkdir(parents=True, exist_ok=True)

    files = {
        "summary": OUT / "task33_2_loss_scenarios.csv",
        "states": OUT / "task33_2_by_state.csv",
        "simulation": OUT / "task34_loan_inputs.csv",
        "assumptions": OUT / "task33_2_assumptions.json",
    }

    summary.to_csv(files["summary"], index=False)
    by_state.to_csv(files["states"], index=False)
    loans.to_csv(files["simulation"], index=False)

    assumptions = {
        "snapshot": "2019-01",
        "pd_event": "First 90+ DPD within 12 months",
        "q_definition": (
            "Credit resolution codes 02/03/09 within "
            "60 months following first 90+ DPD"
        ),
        "q_numerator": HIST_CREDIT_RESOLUTION,
        "q_denominator": HIST_FIRST90,
        "q_historical": q_hist,
        "lgd_historical": HIST_LGD,
        "scenario_multipliers": SCENARIOS,
        "exposure": "As-of UPB, not default-time EAD",
        "loss_horizon": (
            "Potential eventual loss associated with "
            "12M First90 events; not 12M booked losses"
        ),
        "limitations": [
            "Historical q and LGD samples are not "
            "verified as identical cohorts.",
            "Historical q is a 60M conversion measure, "
            "not a lifetime conversion probability.",
            "2019 portfolio calibration is unverified.",
            "PD overpredicts observed First90 events.",
            "Not a validated IFRS9 or CECL ECL model.",
            "No loan dependence or macro stress modeled."
        ],
    }

    files["assumptions"].write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8"
    )

    print("\n=== SCENARIO RESULTS ===")
    print(
        summary[[
            "scenario",
            "q_assumed",
            "loss_proxy_usd",
            "loss_pct_upb"
        ]].to_string(index=False)
    )

    print("\n=== BY STARTING STATE ===")
    print(
        by_state[[
            "scenario",
            "state_t",
            "loans",
            "exposure",
            "loss_proxy"
        ]].to_string(index=False)
    )

    print("\n=== OUTPUT FILES ===")
    for path in files.values():
        print(path)

    # Google Drive persistence
    drive = Path("/content/drive/MyDrive")

    if drive.exists():

        backup = drive / "creditport-task33-backup"
        backup.mkdir(parents=True, exist_ok=True)

        for file in files.values():
            shutil.copy2(file, backup / file.name)

        shutil.copy2(
            Path(__file__),
            backup / Path(__file__).name
        )

        print("\nGoogle Drive backup:", backup)

    else:
        print("\nWARNING: Google Drive not mounted.")


if __name__ == "__main__":
    main()
