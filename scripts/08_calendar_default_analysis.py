from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EVENTS = ROOT / "data/processed/03_loan_events_2006_2022.parquet"
HORIZON = ROOT / "data/processed/07_horizon_events.parquet"

OUT = ROOT / "outputs"
FIG = OUT / "figures"

# March 2026
CUTOFF = 2026 * 12 + 3 - 1

EXPECTED_ELIGIBLE = 818_434


def sql_path(path):
    return str(path).replace("'", "''")


# ============================================================
# Month conversion
# ============================================================

def parse_month(series, name):
    """
    Convert YYYYMM (or an ISO date) to an integer month index.

    Missing values become -1.

    Month index:
        year * 12 + month - 1
    """

    raw = series.astype("string").str.strip()

    missing = raw.isna() | raw.str.lower().isin(
        [
            "",
            "none",
            "nan",
            "nat",
            "null",
            "<na>",
            "0",
            "000000",
        ]
    ).fillna(False)

    six_digit = raw.str.fullmatch(
        r"\d{6}(?:\.0+)?"
    ).fillna(False)

    result = np.full(
        len(raw),
        -1,
        dtype=np.int32,
    )

    if six_digit.any():

        vals = raw.loc[six_digit].str[:6].astype(np.int64)

        year = (vals // 100).to_numpy()
        month = (vals % 100).to_numpy()

        invalid = (
            (year < 1900)
            | (year > 2100)
            | (month < 1)
            | (month > 12)
        )

        if invalid.any():
            raise ValueError(
                f"Invalid YYYYMM values in {name}"
            )

        result[
            np.flatnonzero(six_digit.to_numpy())
        ] = year * 12 + month - 1

    other = ~(missing | six_digit)

    if other.any():

        parsed = pd.to_datetime(
            raw.loc[other],
            errors="coerce",
        )

        if parsed.isna().any():

            examples = (
                raw.loc[other]
                .loc[parsed.isna()]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"Unrecognized {name} dates: {examples}"
            )

        result[
            np.flatnonzero(other.to_numpy())
        ] = (
            parsed.dt.year.to_numpy() * 12
            + parsed.dt.month.to_numpy()
            - 1
        )

    return result


def month_labels(months):

    return [
        f"{m // 12:04d}-{m % 12 + 1:02d}"
        for m in months
    ]


# ============================================================
# Load eligible cohort
# ============================================================

def checked_inputs():

    for file in [EVENTS, HORIZON]:

        if not file.is_file():
            raise FileNotFoundError(file)

    con = duckdb.connect()

    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=2")

    query = f"""
        WITH eligible AS (

            SELECT DISTINCT loan_id

            FROM read_parquet('{sql_path(HORIZON)}')

            WHERE TRY_CAST(horizon AS INTEGER) = 12
              AND eligibility = 'Eligible'
        )

        SELECT
            e.loan_id,
            e.vintage,
            e.first_payment,
            e.last_period,
            e.first_90_period,
            e.zero_effective

        FROM read_parquet('{sql_path(EVENTS)}') AS e

        INNER JOIN eligible AS c
            USING (loan_id)
    """

    try:

        df = con.execute(query).fetchdf()

    except Exception as exc:

        print("\nExpected source columns:")

        print(
            "03: loan_id, vintage, first_payment, "
            "last_period, first_90_period, zero_effective"
        )

        print(
            "07: loan_id, horizon, eligibility"
        )

        raise RuntimeError(
            "Input schema did not match the prior scripts"
        ) from exc

    finally:
        con.close()

    if len(df) != EXPECTED_ELIGIBLE:

        raise ValueError(
            f"Eligible cohort mismatch: "
            f"got {len(df):,}, "
            f"expected {EXPECTED_ELIGIBLE:,}. "
            "Check Task 06 outputs."
        )

    if (
        df["loan_id"].isna().any()
        or df["loan_id"].duplicated().any()
    ):

        raise ValueError(
            "Eligible cohort contains missing/duplicate loan IDs"
        )

    if df["vintage"].isna().any():

        raise ValueError(
            "Missing vintage in eligible cohort"
        )

    df["vintage"] = pd.to_numeric(
        df["vintage"],
        errors="raise",
    ).astype(int)

    if set(df["vintage"].unique()) != set(range(2006, 2023)):

        raise ValueError(
            "Expected exactly the 2006-2022 vintages"
        )

    return df


# ============================================================
# Main analysis
# ============================================================

def main():

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = checked_inputs()

    entry = parse_month(
        df["first_payment"],
        "first_payment",
    )

    last = parse_month(
        df["last_period"],
        "last_period",
    )

    first90 = parse_month(
        df["first_90_period"],
        "first_90_period",
    )

    zero = parse_month(
        df["zero_effective"],
        "zero_effective",
    )

    # --------------------------------------------------------
    # Validate time intervals
    # --------------------------------------------------------

    if ((entry < 0) | (last < 0)).any():

        raise ValueError(
            "Eligible loans must have "
            "first_payment and last_period"
        )

    if (entry > CUTOFF).any() or (last < entry).any():

        raise ValueError(
            "Invalid observation intervals in eligible cohort"
        )

    if ((first90 >= 0) & (first90 < entry)).any():

        raise ValueError(
            "Some eligible loans reached "
            "first 90+ before time origin"
        )

    if ((zero >= 0) & (zero < entry)).any():

        raise ValueError(
            "Some eligible loans exited before time origin"
        )

    # --------------------------------------------------------
    # Define observable risk window
    # --------------------------------------------------------

    observation_end = np.minimum(
        last,
        CUTOFF,
    )

    exit_end = np.where(
        zero >= 0,
        zero,
        CUTOFF,
    )

    exposure_end = np.minimum(
        observation_end,
        exit_end,
    )

    if (exposure_end < entry).any():

        raise ValueError(
            "Some eligible loans have no observable risk month"
        )

    has90 = first90 >= 0

    valid90 = (
        has90
        & (first90 >= entry)
        & (first90 <= exposure_end)
    )

    # The event month is included in the risk set.
    risk_end = np.where(
        valid90,
        first90,
        exposure_end,
    )

    # --------------------------------------------------------
    # Audit timing anomalies
    # --------------------------------------------------------

    audit = {

        "eligible_loans":
            len(df),

        "first90_reported_anytime":
            int(has90.sum()),

        "first90_counted_in_observed_risk_window":
            int(valid90.sum()),

        "first90_after_last_report":
            int(
                (has90 & (first90 > last)).sum()
            ),

        "first90_after_zero_effective":
            int(
                (
                    has90
                    & (zero >= 0)
                    & (first90 > zero)
                ).sum()
            ),

        "first90_same_month_as_zero_effective":
            int(
                (
                    valid90
                    & (zero >= 0)
                    & (first90 == zero)
                ).sum()
            ),

        "zero_effective_after_last_report":
            int(
                (
                    (zero >= 0)
                    & (zero > last)
                ).sum()
            ),

        "zero_effective_before_last_report":
            int(
                (
                    (zero >= 0)
                    & (zero < last)
                ).sum()
            ),

        "observations_censored_at_cutoff":
            int(
                (
                    (zero < 0)
                    & (last == CUTOFF)
                    & ~valid90
                ).sum()
            ),

        "risk_interval_months_total":
            int(
                (risk_end - entry + 1).sum()
            ),
    }

    pd.DataFrame(
        list(audit.items()),
        columns=["check", "count"],
    ).to_csv(
        OUT / "08_risk_set_audit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Build calendar risk sets
    # --------------------------------------------------------

    vintages = np.arange(2006, 2023)

    vi = (
        df["vintage"].to_numpy(dtype=np.int32)
        - vintages[0]
    )

    first_month = int(entry.min())

    calendar = np.arange(
        first_month,
        CUTOFF + 1,
        dtype=np.int32,
    )

    t = len(calendar)

    start_idx = entry - first_month
    end_idx = risk_end - first_month

    # Difference-array method:
    # +1 at entry month
    # -1 immediately after terminal month

    differences = np.zeros(
        (len(vintages), t + 1),
        dtype=np.int64,
    )

    np.add.at(
        differences,
        (vi, start_idx),
        1,
    )

    np.add.at(
        differences,
        (vi, end_idx + 1),
        -1,
    )

    at_risk = np.cumsum(
        differences[:, :t],
        axis=1,
    )

    # --------------------------------------------------------
    # Count first-90 events
    # --------------------------------------------------------

    events = np.zeros(
        (len(vintages), t),
        dtype=np.int64,
    )

    np.add.at(
        events,
        (
            vi[valid90],
            first90[valid90] - first_month,
        ),
        1,
    )

    # --------------------------------------------------------
    # Reconciliation checks
    # --------------------------------------------------------

    if (
        (at_risk < 0).any()
        or (events > at_risk).any()
    ):

        raise AssertionError(
            "Event counts cannot exceed monthly risk sets"
        )

    if int(events.sum()) != int(valid90.sum()):

        raise AssertionError(
            "First 90+ events were lost or double counted"
        )

    if (
        int(at_risk.sum())
        != audit["risk_interval_months_total"]
    ):

        raise AssertionError(
            "Risk-set loan-months failed reconciliation"
        )

    # --------------------------------------------------------
    # Monthly pooled results
    # --------------------------------------------------------

    pooled_risk = at_risk.sum(axis=0)
    pooled_events = events.sum(axis=0)

    monthly = pd.DataFrame({

        "month": month_labels(calendar),

        "year": calendar // 12,

        "at_risk": pooled_risk,

        "first90_events": pooled_events,
    })

    monthly["event_rate_pct"] = np.divide(
        pooled_events * 100.0,
        pooled_risk,
        out=np.full(t, np.nan),
        where=pooled_risk > 0,
    )

    # Weighted 3-month moving event rate
    r3 = (
        monthly["at_risk"]
        .rolling(3, center=True, min_periods=2)
        .sum()
    )

    d3 = (
        monthly["first90_events"]
        .rolling(3, center=True, min_periods=2)
        .sum()
    )

    monthly["weighted_3m_rate_pct"] = (
        d3 / r3 * 100.0
    )

    monthly.to_csv(
        OUT / "08_monthly_first90.csv",
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Annual pooled results
    # --------------------------------------------------------

    annual = monthly.groupby(
        "year",
        as_index=False,
    ).agg(

        first90_events=("first90_events", "sum"),

        risk_loan_months=("at_risk", "sum"),

        observed_months=("month", "count"),
    )

    annual["per_loan_month_rate_pct"] = (
        annual["first90_events"]
        / annual["risk_loan_months"]
        * 100.0
    )

    annual.to_csv(
        OUT / "08_annual_first90.csv",
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Vintage x Calendar Month
    # --------------------------------------------------------

    flat = pd.DataFrame({

        "vintage": np.repeat(vintages, t),

        "month": np.tile(
            monthly["month"].to_numpy(),
            len(vintages),
        ),

        "year": np.tile(
            monthly["year"].to_numpy(),
            len(vintages),
        ),

        "at_risk": at_risk.ravel(),

        "first90_events": events.ravel(),
    })

    flat = flat.loc[
        flat["at_risk"] > 0
    ].copy()

    flat["event_rate_pct"] = (
        flat["first90_events"]
        / flat["at_risk"]
        * 100.0
    )

    flat.to_csv(
        OUT / "08_vintage_monthly_first90.csv",
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Vintage x Calendar Year
    # --------------------------------------------------------

    vintage_year = flat.groupby(
        ["vintage", "year"],
        as_index=False,
    ).agg(

        first90_events=("first90_events", "sum"),

        risk_loan_months=("at_risk", "sum"),
    )

    vintage_year["per_loan_month_rate_pct"] = (
        vintage_year["first90_events"]
        / vintage_year["risk_loan_months"]
        * 100.0
    )

    vintage_year.to_csv(
        OUT / "08_vintage_year_first90.csv",
        index=False,
        float_format="%.6f",
    )

    # ========================================================
    # Figures
    # ========================================================

    plt.rcParams.update({
        "font.size": 10,
    })

    dates = pd.to_datetime(
        monthly["month"] + "-01"
    )

    # --------------------------------------------------------
    # Figure 1: Full calendar-time rate
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 4.8)
    )

    ax.plot(
        dates,
        monthly["event_rate_pct"],
        alpha=0.33,
        lw=0.8,
        label="Monthly",
    )

    ax.plot(
        dates,
        monthly["weighted_3m_rate_pct"],
        lw=1.8,
        label="Weighted 3-month rate",
    )

    ax.set(
        title="First 90+ DPD: reported calendar month",
        ylabel="First 90+ events / at-risk loans (%)",
        xlabel="Calendar month",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        FIG / "08_calendar_monthly_rate.png",
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: Focus windows
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(11, 7),
    )

    focus_windows = [
        (axs[0], "2007-01", "2009-12"),
        (axs[1], "2019-01", "2021-12"),
    ]

    for ax, lo, hi in focus_windows:

        m = (
            (monthly["month"] >= lo)
            & (monthly["month"] <= hi)
        )

        ax.plot(
            dates[m],
            monthly.loc[m, "event_rate_pct"],
            alpha=0.35,
            lw=1,
            label="Monthly",
        )

        ax.plot(
            dates[m],
            monthly.loc[m, "weighted_3m_rate_pct"],
            lw=2,
            label="Weighted 3-month",
        )

        ax.set(
            title=f"{lo[:4]}-{hi[:4]}",
            ylabel="Event rate (%)",
        )

        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    axs[-1].set_xlabel("Calendar month")

    fig.suptitle(
        "Focus windows: pooled at-risk-normalized "
        "first 90+ rates"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "08_calendar_focus_windows.png",
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Figure 3: Vintage x Calendar Year heatmap
    # --------------------------------------------------------

    years = np.arange(
        int(calendar[0] // 12),
        2027,
    )

    heat = np.full(
        (len(vintages), len(years)),
        np.nan,
    )

    for row in vintage_year.itertuples(index=False):

        # Sparse exposures are hidden in the figure only.
        if row.risk_loan_months >= 1000:

            heat[
                row.vintage - 2006,
                row.year - years[0],
            ] = row.per_loan_month_rate_pct

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
    )

    ax.set_xticks(
        np.arange(len(years)),
        labels=years,
        rotation=60,
    )

    ax.set_yticks(
        np.arange(len(vintages)),
        labels=vintages,
    )

    ax.set(
        xlabel="Calendar year",
        ylabel="Origination vintage",
        title="First 90+ events per at-risk loan-month (%)",
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Monthly rate (%)",
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "08_vintage_calendar_heatmap.png",
        dpi=160,
    )

    plt.close(fig)

    # ========================================================
    # Short terminal summary
    # ========================================================

    print(f"Eligible loans: {len(df):,}")

    print(
        "Observed first-90 events: "
        f"{int(valid90.sum()):,}"
    )

    print(
        "Risk loan-months: "
        f"{int(at_risk.sum()):,}"
    )

    print("\nRisk-set audit (selected):")

    keys = [
        "first90_after_last_report",
        "first90_after_zero_effective",
        "first90_same_month_as_zero_effective",
        "zero_effective_after_last_report",
    ]

    for key in keys:
        print(f"  {key}: {audit[key]:,}")

    print(
        "\nFocus-year summary "
        "(pooled; % per at-risk loan-month):"
    )

    selected = annual.loc[
        annual["year"].isin(
            [2007, 2008, 2009, 2019, 2020, 2021]
        ),
        [
            "year",
            "first90_events",
            "risk_loan_months",
            "per_loan_month_rate_pct",
        ],
    ]

    print(
        selected.round(4).to_string(index=False)
    )

    print(
        "\nSaved 5 CSV files and 3 figures "
        "under outputs/."
    )


if __name__ == "__main__":
    main()