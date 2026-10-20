#!/usr/bin/env python3
"""Generate four GitHub README figures, with no raw mortgage data required.

Run from the repository root:
    python3 scripts/37_make_readme_figures.py

The model metrics come from the published model validation tables. The
M3 benchmark numbers are from the previously reported Task 31 validation.
Portfolio metrics are read from reports/tables, when present, and checked
against the reported project values. This generates documentation only;
it does not run models or overwrite the original Task 34/35 figures.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#11243A"
BLUE = "#2B5274"
TEAL = "#137F7A"
GOLD = "#BD8532"
INK = "#24364A"
MUTED = "#66788A"
LINE = "#DCE5EA"
BG = "#FFFFFF"
PALE = "#F4F8FA"
PALE_TEAL = "#E6F3F0"
PALE_BLUE = "#EBF2FA"
PALE_AMBER = "#FAF2E3"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=175, bbox_inches="tight", facecolor=BG, pad_inches=0.15)
    plt.close(fig)
    print(f"Created: {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def card(ax, center, width, height, heading, detail, face=PALE, edge=LINE,
         titlecolor=NAVY, subsize=10.1):
    x, y = center
    rect = FancyBboxPatch(
        (x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        linewidth=1.15, edgecolor=edge, facecolor=face, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.25, heading, ha="center", va="center",
            fontsize=12.8, weight="bold", color=titlecolor, zorder=4)
    ax.text(x, y - 0.24, detail, ha="center", va="center",
            fontsize=subsize, color=MUTED, linespacing=1.4, zorder=4)


def arrow(ax, start, end, color="#7892A2", curved=0, lw=2.1):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                        linewidth=lw, color=color, zorder=2,
                        connectionstyle=f"arc3,rad={curved}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def title(ax, heading, subtitle):
    ax.text(0.45, 8.50, heading, fontsize=23, weight="bold", color=NAVY, va="center")
    ax.text(0.47, 8.08, subtitle, fontsize=11, color=MUTED, va="center")


def workflow():
    fig, ax = plt.subplots(figsize=(15.8, 8.3))
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    title(ax, "Mortgage credit risk | End-to-end workflow",
          "Loan-level data quality  ->  First90 probabilities  ->  historical loss references  ->  portfolio risk")

    top_x = [1.95, 5.65, 9.35, 13.05]
    top_y = 6.42
    labels = [
        ("Freddie Mac", "850k sampled loans\n51.7m monthly records"),
        ("Loan-level QA", "Observation windows\nExit & loss accounting"),
        ("Monthly models", "Credit-state LightGBM\nNo-macro / With-macro"),
        ("12-month First90", "64 simulated paths / loan\n30k-loan 2019 snapshot"),
    ]
    for i, (h, d) in enumerate(labels):
        card(ax, (top_x[i], top_y), 3.10, 1.45, h, d,
             face=PALE_BLUE if i in (0, 2) else PALE, subsize=9.4)
    for i in range(3):
        arrow(ax, (top_x[i]+1.61, top_y), (top_x[i+1]-1.65, top_y))

    card(ax, (4.05, 4.20), 4.45, 1.52, "Historical loss audits",
         "First90-to-resolution conversion (q)\nDisposition-sample LGD",
         face=PALE_AMBER)
    card(ax, (11.72, 4.20), 4.45, 1.52, "Exposure proxy",
         "12M First90 PD  x  as-of UPB", face=PALE_TEAL)

    arrow(ax, (5.65, 5.68), (4.50, 4.99), color=GOLD, curved=-0.13)
    arrow(ax, (13.05, 5.69), (12.10, 4.99), color=TEAL, curved=0.08)

    card(ax, (4.25, 1.76), 3.78, 1.44, "Loss proxy scenarios",
         "PD  x  UPB  x  q  x  LGD", face=PALE_TEAL)
    card(ax, (9.18, 1.76), 4.16, 1.44, "Portfolio loss simulation",
         "Independent vs assumed common factor\n99% VaR / expected shortfall",
         face=PALE_BLUE, subsize=9.4)
    card(ax, (13.60, 1.76), 3.70, 1.44, "Stress sensitivity",
         "PD / q / LGD shocks\nCombined scenario",
         face=PALE_AMBER, subsize=9.4)

    arrow(ax, (4.05, 3.41), (4.24, 2.51), color=GOLD, curved=-0.06)
    arrow(ax, (11.72, 3.41), (5.54, 2.50), color=TEAL, curved=-0.01)
    arrow(ax, (6.17, 1.76), (7.05, 1.76), color=TEAL)
    arrow(ax, (11.32, 1.76), (11.70, 1.76), color=GOLD)

    ax.text(0.53, 0.45,
            "Model-risk checks throughout: as-of integrity  |  censoring and cohort alignment  |  state-level calibration  |  assumed dependence",
            color=MUTED, fontsize=10.6)
    save(fig, "project_workflow.png")


def credit_states():
    fig, ax = plt.subplots(figsize=(15.6, 6.9))
    ax.set(xlim=(0, 16), ylim=(0, 8))
    ax.axis("off")
    ax.text(0.42, 7.48, "Credit-state framework", fontsize=23,
            weight="bold", color=NAVY)
    ax.text(0.44, 7.06, "Eight modeled states: five observed active states and three loan-exit categories",
            fontsize=11, color=MUTED)

    ax.text(0.56, 6.31, "ACTIVE / NON-TERMINAL STATES", color=BLUE,
            fontsize=10.5, weight="bold")
    ax.text(0.56, 3.37, "EXIT CATEGORIES", color=TEAL,
            fontsize=10.5, weight="bold")
    centers = [1.80, 4.88, 7.96, 11.04, 14.12]
    active = [
        ("S0", "Current"),
        ("S1", "30-59 DPD"),
        ("S2", "60-89 DPD"),
        ("S3", "90+ DPD"),
        ("S4", "Foreclosure"),
    ]
    for i, ((state, name), cx) in enumerate(zip(active, centers)):
        face = PALE_TEAL if i == 0 else (PALE_AMBER if i >= 3 else PALE_BLUE)
        rect = FancyBboxPatch((cx-1.20, 4.60), 2.40, 1.32,
                             boxstyle="round,pad=0.04,rounding_size=0.17",
                             facecolor=face, edgecolor=LINE, linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, 5.43, state, ha="center", va="center", weight="bold",
                fontsize=17.5, color=NAVY, zorder=4)
        ax.text(cx, 4.98, name, ha="center", va="center",
                fontsize=10.9, color=INK, zorder=4)
    for cx1, cx2 in zip(centers[:-1], centers[1:]):
        arrow(ax, (cx1+1.21, 5.26), (cx2-1.23, 5.26), color="#7892A2", lw=1.9)
    exits = [("S5", "Payoff"), ("S6", "Credit resolution"), ("S7", "Other exit")]
    ex = [3.28, 8.0, 12.72]
    for (s, d), x in zip(exits, ex):
        rect = FancyBboxPatch((x-1.85, 1.75), 3.70, 1.12,
                             boxstyle="round,pad=0.05,rounding_size=0.17",
                             facecolor=PALE_TEAL, edgecolor=LINE,
                             linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, 2.36, f"{s}  |  {d}", fontsize=12.2,
                weight="bold", color=NAVY, ha="center", va="center")
    ax.text(0.55, 0.70,
            "Arrows illustrate increasing delinquency severity only. Cures, roll-backs, and exits are modeled from observed data; this is not the full transition matrix.",
            fontsize=10.3, color=MUTED, va="center")
    save(fig, "credit_state_framework.png")


def lookup_csv(file, criteria):
    if not file.is_file():
        raise FileNotFoundError(f"Missing source table: {file}")
    with file.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one match in {file}: {criteria}; found {len(matches)}")
    return matches[0]


def model_comparison():
    val = ROOT / "models"
    no = lookup_csv(val / "nomacro/validation_breakdown.csv", {"group": "all"})
    yes = lookup_csv(val / "withmacro/validation_breakdown.csv", {"group": "all"})
    names = ["Logistic M3", "LightGBM\nNo macro", "LightGBM\nWith macro"]
    logloss = [0.12760, float(no["logloss"]), float(yes["logloss"])]
    brier = [0.05007, float(no["brier"]), float(yes["brier"])]
    oe = [1.15716, float(no["first90_OE"]), float(yes["first90_OE"])]

    fig = plt.figure(figsize=(15.7, 6.1))
    fig.text(0.035, 0.925, "Monthly state-model comparison", fontsize=23,
             weight="bold", color=NAVY)
    fig.text(0.035, 0.867,
             "Development evaluation (2018-2020) | the first year was used for early stopping; not independent OOT",
             fontsize=11, color=MUTED)
    axes = [fig.add_axes(p) for p in [(0.07,0.30,0.25,0.43), (0.39,0.30,0.25,0.43), (0.72,0.30,0.25,0.43)]]
    vals = [logloss, brier, oe]
    captions = ["Multiclass log loss", "Brier score", "First90 observed / expected"]
    hints = ["LOWER IS BETTER", "LOWER IS BETTER", "CLOSER TO 1 IS BETTER"]
    colors = [BLUE, "#5F98A0", TEAL]

    for j, (ax, data, label, hint) in enumerate(zip(axes, vals, captions, hints)):
        y = np.array([2,1,0])
        bars = ax.barh(y, data, color=colors, height=0.51)
        for k, (bar, v) in enumerate(zip(bars, data)):
            ax.text(v + (0.002 if j == 0 else 0.0008 if j == 1 else 0.024),
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}" if j < 2 else f"{v:.3f}",
                    va="center", ha="left", fontsize=10.8, weight="bold", color=NAVY)
        ax.set_yticks(y, names, fontsize=10.4)
        ax.set_ylim(-0.7, 2.7)
        ax.set_xlim(0, [0.161, 0.063, 1.46][j])
        ax.set_title(label, fontsize=13.4, weight="bold", pad=18, color=NAVY)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=9.5, pad=6)
        if j == 2:
            ax.axvline(1.0, color=GOLD, linestyle="--", linewidth=1.6, alpha=0.9)
        ax.text(0.5, -0.26, hint, fontsize=9.3, color=MUTED,
                transform=ax.transAxes, ha="center", weight="bold")
    fig.text(0.035, 0.065,
             "Source: saved model validation tables; Logistic M3 from Task 31 validation. Overall log loss does not establish state-level calibration.",
             fontsize=10.2, color=MUTED)
    save(fig, "model_comparison.png")


def risk_snapshot():
    task33 = lookup_csv(
        ROOT / "reports/tables/task33_2_loss_scenarios.csv",
        {"scenario": "Historical Reference"},
    )
    task34 = lookup_csv(
        ROOT / "reports/tables/task34_summary.csv",
        {"scenario": "Historical Reference", "latent_factor_rho_assumed": "0.15"},
    )
    task35 = lookup_csv(
        ROOT / "reports/tables/task35_stress_summary.csv",
        {"scenario": "Combined Stress"},
    )
    upb = float(task33["total_upb_usd"])
    exposure = float(task33["first90_exposure_usd"])
    expected = float(task33["expected_first90_events"])
    loss = float(task33["loss_proxy_usd"])
    var = float(task34["VaR_99_usd"])
    stress = float(task35["loss_proxy_usd"])
    assert 5.4e9 < upb < 5.6e9
    assert 27e6 < exposure < 30e6
    assert 5e6 < loss < 6e6

    fig, ax = plt.subplots(figsize=(15.7, 7.35))
    ax.set(xlim=(0, 16), ylim=(0, 8))
    ax.axis("off")
    ax.text(0.44, 7.43, "Portfolio risk | Key metrics", fontsize=24,
            weight="bold", color=NAVY)
    ax.text(0.44, 7.02, "January 2019 as-of snapshot | 30,000-loan sample | 12-month First90 horizon",
            fontsize=11.5, color=MUTED)
    metrics = [
        ("PORTFOLIO UPB", f"${upb/1e9:.2f}B", "As-of unpaid principal balance", PALE_BLUE),
        ("EXPECTED FIRST90", f"{expected:.1f}", "111 observed in development sample", PALE),
        ("PD-WEIGHTED EXPOSURE", f"${exposure/1e6:.2f}M", "12M First90 PD  x  as-of UPB", PALE_TEAL),
        ("LOSS REFERENCE PROXY", f"${loss/1e6:.2f}M", "Historical conversion q and LGD", PALE_TEAL),
        ("99% VaR | SHARED FACTOR", f"${var/1e6:.2f}M", "Assumed Gaussian latent rho = 0.15", PALE_AMBER),
        ("COMBINED STRESS", f"${stress/1e6:.2f}M", "Illustrative PD / q / LGD shocks", PALE_AMBER),
    ]
    xloc = [2.88, 8.0, 13.12]
    yloc = [4.91, 2.50]
    for idx, (label, value, desc, face) in enumerate(metrics):
        x, y = xloc[idx % 3], yloc[idx // 3]
        patch = FancyBboxPatch((x-2.27, y-1.08), 4.54, 2.10,
                               boxstyle="round,pad=0.04,rounding_size=0.20",
                               facecolor=face, edgecolor=LINE, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x, y+0.56, label, ha="center", va="center",
                fontsize=10.4, weight="bold", color=BLUE)
        ax.text(x, y+0.08, value, ha="center", va="center",
                fontsize=29, weight="bold", color=NAVY)
        ax.text(x, y-0.55, desc, ha="center", va="center",
                fontsize=9.7, color=MUTED)
    ax.text(0.54, 0.56,
            "Illustrative analysis only: loss is a conditional historical-reference proxy; VaR depends on assumed correlation; not validated ECL or economic capital.",
            color=MUTED, fontsize=10.2)
    save(fig, "risk_metrics_snapshot.png")


def main():
    workflow()
    credit_states()
    model_comparison()
    risk_snapshot()
    print("Finished: four README figures created. Existing Task 34/35 figures untouched.")

if __name__ == "__main__":
    main()
