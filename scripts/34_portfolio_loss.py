
from pathlib import Path
import argparse
import json
import math
import shutil

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT / "outputs/modeling/task33_loss"
    / "task34_loan_inputs.csv"
)

OUT = ROOT / "outputs/modeling/task34_portfolio_loss"

# Previously audited historical references
HIST_Q = 10396 / 24324
HIST_LGD = 0.4645

Q_MULT = {
    "Lower": 0.75,
    "Historical Reference": 1.00,
    "Higher": 1.25,
}

# Illustrative latent-factor correlations.
# These are NOT estimated mortgage default correlations.
FACTOR_RHOS = (0.00, 0.15)


def summarize(loss, events, resolutions,
              scenario, q, rho, analytic_el):

    n = len(loss)
    sorted_loss = np.sort(loss)

    mean = float(np.mean(loss))
    sigma = float(np.std(loss, ddof=1))

    result = {
        "scenario": scenario,
        "latent_factor_rho_assumed": rho,
        "q_assumed": q,
        "lgd_fixed": HIST_LGD,
        "simulations": n,
        "analytic_expected_loss_usd": analytic_el,
        "simulated_mean_loss_usd": mean,
        "simulation_SE_mean_usd": sigma / np.sqrt(n),
        "unexpected_loss_std_usd": sigma,
        "mean_first90_loans": float(np.mean(events)),
        "mean_credit_resolutions": float(
            np.mean(resolutions)
        ),
    }

    for alpha in (0.95, 0.99):

        k = max(
            1,
            int(math.ceil((1 - alpha) * n - 1e-9))
        )

        var = float(
            np.quantile(loss, alpha, method="higher")
        )

        es = float(np.mean(sorted_loss[-k:]))

        level = int(alpha * 100)

        result[f"VaR_{level}_usd"] = var
        result[f"ES_{level}_usd"] = es

        result[f"VaR_{level}_minus_EL_usd"] = (
            var - mean
        )

    return result


def simulate(pd12, upb, n_sim, batch_size, seed):

    rng = np.random.default_rng(seed)

    thresholds = ndtri(pd12)
    loss_weight = upb * HIST_LGD

    qs = {
        name: min(1.0, HIST_Q * multiplier)
        for name, multiplier in Q_MULT.items()
    }

    expected_exposure = float(np.dot(pd12, upb))

    summaries = []
    all_draws = []
    plot_data = {}

    for rho in FACTOR_RHOS:

        print(
            f"\nRunning latent rho={rho:.2f} ...",
            flush=True
        )

        losses = {
            name: np.empty(n_sim, dtype=np.float64)
            for name in qs
        }

        resolutions = {
            name: np.empty(n_sim, dtype=np.int32)
            for name in qs
        }

        event_counts = np.empty(
            n_sim, dtype=np.int32
        )

        for start in range(0, n_sim, batch_size):

            b = min(batch_size, n_sim - start)

            # Systematic Gaussian factor
            z = rng.standard_normal(b)

            if rho == 0.0:

                p_cond = pd12[None, :]

            else:

                # One-factor Gaussian dependence.
                # Preserves each loan's marginal 12M PD.
                p_cond = ndtr(
                    (
                        thresholds[None, :]
                        - np.sqrt(rho) * z[:, None]
                    )
                    / np.sqrt(1 - rho)
                )

            # Simulate First90 events
            events = (
                rng.random(
                    (b, len(pd12)),
                    dtype=np.float32
                )
                < p_cond
            )

            # Independent conditional conversion draws.
            # Reused across q scenarios for comparability.
            conversion_u = rng.random(
                (b, len(pd12)),
                dtype=np.float32
            )

            event_counts[start:start + b] = (
                events.sum(axis=1)
            )

            for name, q in qs.items():

                resolved = events & (conversion_u < q)

                resolutions[name][start:start + b] = (
                    resolved.sum(axis=1)
                )

                losses[name][start:start + b] = (
                    np.einsum(
                        "ij,j->i",
                        resolved,
                        loss_weight,
                        optimize=False
                    )
                )

        for name, q in qs.items():

            loss = losses[name]

            analytic_el = (
                expected_exposure * q * HIST_LGD
            )

            stats = summarize(
                loss,
                event_counts,
                resolutions[name],
                name,
                q,
                rho,
                analytic_el
            )

            summaries.append(stats)

            # Monte Carlo convergence diagnostic
            difference = abs(
                stats["simulated_mean_loss_usd"]
                - analytic_el
            )

            if difference > (
                5 * stats["simulation_SE_mean_usd"]
            ):
                print(
                    "WARNING: Monte Carlo mean differs "
                    "from analytical expectation by >5 SE:",
                    name, rho
                )

            all_draws.append(
                pd.DataFrame({
                    "simulation_id": np.arange(n_sim),
                    "scenario": name,
                    "latent_factor_rho_assumed": rho,
                    "first90_count": event_counts,
                    "credit_resolution_count":
                        resolutions[name],
                    "loss_proxy_usd": loss,
                })
            )

            if name == "Historical Reference":
                plot_data[rho] = loss

    summary = pd.DataFrame(summaries)
    draws = pd.concat(all_draws, ignore_index=True)

    return summary, draws, plot_data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simulations",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=128
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261020
    )

    args = parser.parse_args()

    if args.simulations < 1000 or args.batch < 1:
        raise ValueError(
            "Use >=1000 simulations and batch >=1"
        )

    if not SRC.exists():
        raise FileNotFoundError(SRC)

    df = pd.read_csv(SRC, low_memory=False)

    required = [
        "pd_12m",
        "upb_asof",
        "state_t"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) != 30000:
        raise ValueError(
            f"Expected 30,000 loans, got {len(df)}"
        )

    pd12 = pd.to_numeric(
        df["pd_12m"],
        errors="raise"
    ).to_numpy(dtype=np.float64)

    upb = pd.to_numeric(
        df["upb_asof"],
        errors="raise"
    ).to_numpy(dtype=np.float64)

    if (
        not np.isfinite(pd12).all()
        or not np.isfinite(upb).all()
    ):
        raise ValueError("Non-finite inputs")

    if (
        np.any(pd12 < 0)
        or np.any(pd12 > 1)
        or np.any(upb < 0)
    ):
        raise ValueError("Invalid PD or UPB")

    if not np.isclose(
        pd12.sum(), 155.52, atol=0.30
    ):
        raise ValueError("PD total mismatch")

    exposure = float(np.dot(pd12, upb))

    if not np.isclose(
        exposure,
        28_418_097.39,
        rtol=0.001
    ):
        raise ValueError(
            "Task 33.1 exposure mismatch"
        )

    keys = ["_vintage_file", "loan_id"]

    if set(keys).issubset(df.columns):
        if df.duplicated(keys).any():
            raise ValueError("Duplicate loans")

    print("=== TASK 34: PORTFOLIO LOSS SIMULATION ===")
    print(f"Loans: {len(df):,}")
    print(f"Expected First90: {pd12.sum():.2f}")
    print(f"First90 exposure: ${exposure:,.2f}")
    print(f"Historical q: {HIST_Q:.4%}")
    print(f"Historical LGD: {HIST_LGD:.2%}")

    summary, draws, plot_data = simulate(
        pd12,
        upb,
        args.simulations,
        args.batch,
        args.seed
    )

    OUT.mkdir(parents=True, exist_ok=True)

    f_summary = OUT / "task34_summary.csv"
    f_draws = OUT / "task34_simulation_draws.csv"
    f_assumptions = OUT / "task34_assumptions.json"
    f_chart = OUT / "task34_loss_distribution.png"

    summary.to_csv(f_summary, index=False)
    draws.to_csv(f_draws, index=False)

    assumptions = {
        "snapshot": "2019-01",
        "event": "First 90+ DPD within 12 months",
        "loss_horizon": (
            "Potential eventual loss following 12M "
            "First90 events; not 12M recognized loss"
        ),
        "EAD": (
            "As-of UPB proxy, not actual default-time EAD"
        ),
        "q_historical_60m": HIST_Q,
        "q_multipliers": Q_MULT,
        "LGD_fixed": HIST_LGD,
        "latent_factor_rhos": FACTOR_RHOS,
        "dependence": (
            "One-factor Gaussian dependence for First90. "
            "Rho is latent asset correlation, not observed "
            "default correlation. No empirical calibration."
        ),
        "conversion": (
            "Bernoulli conditional on First90, "
            "independent of common factor"
        ),
        "LGD_model": (
            "Fixed historical disposition LGD; "
            "not a predictive severity distribution"
        ),
        "limitations": [
            "Historical q and LGD cohorts are not "
            "confirmed to match exactly.",
            "First90 PD is overpredicted on the "
            "2019 development sample (O/E 0.7137).",
            "Common-factor correlation is hypothetical.",
            "Conversion and LGD macro dependence omitted.",
            "As-of UPB is not default-time EAD.",
            "Full-release historical estimates are "
            "not confirmed as 2019 as-of available.",
            "VaR/ES are hypothetical scenario results, "
            "not validated economic capital."
        ]
    }

    f_assumptions.write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8"
    )

    # Generate loss distribution chart
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for rho, losses in plot_data.items():

        ax.hist(
            losses / 1e6,
            bins=65,
            density=True,
            alpha=0.45,
            label=f"Latent rho={rho:.2f}"
        )

    ax.set_title(
        "Portfolio Loss Distribution — "
        "Historical Reference Scenario"
    )

    ax.set_xlabel(
        "Potential eventual loss proxy (USD millions)"
    )

    ax.set_ylabel("Simulation density")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(f_chart, dpi=170)
    plt.close(fig)

    # Display results in USD millions
    print("\n=== TASK 34 RESULTS (USD MILLIONS) ===")

    cols = [
        "scenario",
        "latent_factor_rho_assumed",
        "simulations",
        "analytic_expected_loss_usd",
        "simulated_mean_loss_usd",
        "unexpected_loss_std_usd",
        "VaR_95_usd",
        "ES_95_usd",
        "VaR_99_usd",
        "ES_99_usd"
    ]

    view = summary[cols].copy()

    for col in view.columns:
        if col.endswith("_usd"):
            view[col] = (
                view[col] / 1e6
            ).round(3)

    print(view.to_string(index=False))

    print("\n=== MONTE CARLO DIAGNOSTICS ===")

    print(
        summary[[
            "scenario",
            "latent_factor_rho_assumed",
            "mean_first90_loans",
            "mean_credit_resolutions",
            "simulation_SE_mean_usd"
        ]].to_string(index=False)
    )

    print("\n=== OUTPUT FILES ===")
    for path in [
        f_summary,
        f_draws,
        f_assumptions,
        f_chart
    ]:
        print(path)

    # Persist outputs to Google Drive
    drive = Path("/content/drive/MyDrive")

    if drive.exists():

        backup = drive / "creditport-task34-backup"
        backup.mkdir(parents=True, exist_ok=True)

        for path in [
            f_summary,
            f_draws,
            f_assumptions,
            f_chart,
            Path(__file__)
        ]:
            shutil.copy2(
                path,
                backup / path.name
            )

        print("\nGoogle Drive backup:", backup)

    else:
        print("\nWARNING: Google Drive not mounted")


if __name__ == "__main__":
    main()
