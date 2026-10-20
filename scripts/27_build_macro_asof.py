
"""Task 27.5C: Historical as-of macroeconomic feature table.

Uses conservative publication-lag proxies, not archived
real-time data vintages.

Does not modify loan-level data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
MACRO = ROOT / "data/processed/macro"
REPORT = ROOT / "outputs/pd_model"

START = "2006-01-31"
END = "2026-03-31"

# Conservative assumed publication delays.
HPI_LAG_MONTHS = 3
BLS_LAG_MONTHS = 1

MAX_HPI_AGE_MONTHS = 9
MAX_BLS_AGE_MONTHS = 3
MAX_PMMS_AGE_DAYS = 14


# ============================================================
# 1. Utilities
# ============================================================

def get_csv(name, columns):

    path = MACRO / name

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )

    return df


def parse_date(df, col):

    df[col] = (
        pd.to_datetime(df[col], errors="coerce")
        .dt.normalize()
    )

    if df[col].isna().any():
        raise ValueError(f"Invalid dates in {col}")


def month_age(now, past):

    return (
        (now.dt.year - past.dt.year) * 12
        + (now.dt.month - past.dt.month)
    )


# ============================================================
# 2. FHFA house-price index
# ============================================================

def prepare_hpi():

    df = get_csv(
        "27d_fhfa_state_hpi_quarterly.csv",
        ["state", "period_end", "hpi_sa"],
    )

    parse_date(df, "period_end")

    df["state"] = (
        df["state"].astype(str).str.strip().str.upper()
    )

    df["hpi_sa"] = pd.to_numeric(
        df["hpi_sa"], errors="coerce"
    )

    if (
        df["hpi_sa"].isna().any()
        or (df["hpi_sa"] <= 0).any()
    ):
        raise ValueError("Invalid FHFA HPI")

    if df.duplicated(["state", "period_end"]).any():
        raise ValueError("Duplicate state-quarter HPI")

    df = df.sort_values(
        ["state", "period_end"]
    ).copy()

    prior = df.groupby("state")["hpi_sa"].shift(4)

    prev_period = (
        df.groupby("state")["period_end"].shift(4)
    )

    valid_4q = (
        df["period_end"]
        .dt.to_period("Q").astype("int64")
        -
        prev_period
        .dt.to_period("Q").astype("int64")
    ) == 4

    df["hpi_yoy_pct"] = np.where(
        valid_4q,
        (df["hpi_sa"] / prior - 1) * 100,
        np.nan,
    )

    # The quarter becomes eligible three month-ends later.
    df["hpi_available_proxy"] = (
        df["period_end"]
        + pd.offsets.MonthEnd(HPI_LAG_MONTHS)
    )

    return df.rename(columns={
        "period_end": "hpi_observation_end",
        "hpi_sa": "hpi_state_sa",
    })[
        [
            "state",
            "hpi_observation_end",
            "hpi_available_proxy",
            "hpi_state_sa",
            "hpi_yoy_pct",
        ]
    ]


# ============================================================
# 3. BLS unemployment
# ============================================================

def prepare_bls():

    df = get_csv(
        "27d_bls_state_unemployment_monthly.csv",
        ["state", "period_end", "unemployment_pct"],
    )

    parse_date(df, "period_end")

    df["state"] = (
        df["state"].astype(str).str.strip().str.upper()
    )

    df["unemployment_pct"] = pd.to_numeric(
        df["unemployment_pct"],
        errors="coerce",
    )

    if df.duplicated(["state", "period_end"]).any():
        raise ValueError("Duplicate state-month unemployment")

    df = df.sort_values(
        ["state", "period_end"]
    ).copy()

    prior = (
        df.groupby("state")["unemployment_pct"]
        .shift(3)
    )

    prev_period = (
        df.groupby("state")["period_end"]
        .shift(3)
    )

    consecutive = (
        df["period_end"]
        .dt.to_period("M").astype("int64")
        -
        prev_period
        .dt.to_period("M").astype("int64")
    ) == 3

    # Change in unemployment, in percentage points.
    df["unemployment_change_3m_pp"] = (
        df["unemployment_pct"] - prior
    ).where(consecutive)

    df["bls_available_proxy"] = (
        df["period_end"]
        + pd.offsets.MonthEnd(BLS_LAG_MONTHS)
    )

    # Exclude missing observations from eligible source values.
    # The later as-of join will use the latest available
    # earlier VALID reading, with its age recorded explicitly.
    df = df.dropna(
        subset=["unemployment_pct"]
    )

    return df.rename(columns={
        "period_end": "bls_observation_end",
    })[
        [
            "state",
            "bls_observation_end",
            "bls_available_proxy",
            "unemployment_pct",
            "unemployment_change_3m_pp",
        ]
    ]


# ============================================================
# 4. Freddie Mac PMMS
# ============================================================

def prepare_pmms():

    df = get_csv(
        "27d_pmms_mortgage30_weekly.csv",
        ["date", "mortgage30_pct"],
    )

    parse_date(df, "date")

    df["mortgage30_pct"] = pd.to_numeric(
        df["mortgage30_pct"],
        errors="coerce",
    )

    if df["date"].duplicated().any():
        raise ValueError("Duplicate PMMS dates")

    if df["mortgage30_pct"].isna().any():
        raise ValueError("Missing PMMS values")

    return (
        df.rename(
            columns={"date": "pmms_observation_date"}
        )
        .sort_values("pmms_observation_date")
    )


# ============================================================
# 5. Historical as-of alignment
# ============================================================

def main():

    REPORT.mkdir(parents=True, exist_ok=True)

    hpi = prepare_hpi()
    bls = prepare_bls()
    pmms = prepare_pmms()

    states = sorted(bls["state"].unique())

    if (
        len(states) != 51
        or set(states) - set(hpi["state"].unique())
    ):
        raise ValueError(
            "Expected 51 states/DC in both HPI and BLS"
        )

    dates = pd.date_range(
        START, END, freq="ME"
    )

    frames = []

    for state in states:

        frame = pd.DataFrame({
            "state": state,
            "asof_month_end": dates,
        })

        hs = (
            hpi.loc[hpi["state"] == state]
            .drop(columns="state")
            .sort_values("hpi_available_proxy")
        )

        bs = (
            bls.loc[bls["state"] == state]
            .drop(columns="state")
            .sort_values("bls_available_proxy")
        )

        # Most recent HPI legally eligible under
        # the assumed reporting-lag protocol.
        frame = pd.merge_asof(
            frame,
            hs,
            left_on="asof_month_end",
            right_on="hpi_available_proxy",
            direction="backward",
        )

        # Most recent available BLS reading.
        frame = pd.merge_asof(
            frame,
            bs,
            left_on="asof_month_end",
            right_on="bls_available_proxy",
            direction="backward",
        )

        # Latest PMMS publication on/before month-end.
        frame = pd.merge_asof(
            frame,
            pmms,
            left_on="asof_month_end",
            right_on="pmms_observation_date",
            direction="backward",
        )

        # ----------------------------------------------------
        # Freshness diagnostics
        # ----------------------------------------------------

        frame["hpi_age_months"] = month_age(
            frame["asof_month_end"],
            frame["hpi_observation_end"],
        )

        frame["bls_age_months"] = month_age(
            frame["asof_month_end"],
            frame["bls_observation_end"],
        )

        frame["pmms_age_days"] = (
            frame["asof_month_end"]
            - frame["pmms_observation_date"]
        ).dt.days

        frame["bls_stale"] = (
            frame["bls_age_months"].isna()
            | (
                frame["bls_age_months"]
                > MAX_BLS_AGE_MONTHS
            )
        )

        frame["hpi_stale"] = (
            frame["hpi_age_months"].isna()
            | (
                frame["hpi_age_months"]
                > MAX_HPI_AGE_MONTHS
            )
        )

        frame["pmms_stale"] = (
            frame["pmms_age_days"].isna()
            | (
                frame["pmms_age_days"]
                > MAX_PMMS_AGE_DAYS
            )
        )

        frame.loc[
            frame["bls_stale"],
            [
                "unemployment_pct",
                "unemployment_change_3m_pp",
            ],
        ] = np.nan

        frame.loc[
            frame["hpi_stale"],
            [
                "hpi_state_sa",
                "hpi_yoy_pct",
            ],
        ] = np.nan

        frame.loc[
            frame["pmms_stale"],
            "mortgage30_pct",
        ] = np.nan

        frames.append(frame)

    result = pd.concat(
        frames, ignore_index=True
    )

    # --------------------------------------------------------
    # 6. Quality checks
    # --------------------------------------------------------

    if result.duplicated(
        ["state", "asof_month_end"]
    ).any():
        raise ValueError("Duplicate state/month in output")

    if len(result) != 51 * len(dates):
        raise ValueError("Unexpected macro feature row count")

    for source_col in [
        "hpi_available_proxy",
        "bls_available_proxy",
        "pmms_observation_date",
    ]:

        if (
            result[source_col] > result["asof_month_end"]
        ).any():
            raise ValueError(
                f"Future-dated source observation: {source_col}"
            )

    result["asof_yyyymm"] = (
        result["asof_month_end"].dt.year * 100
        + result["asof_month_end"].dt.month
    )

    result = result.sort_values(
        ["state", "asof_month_end"]
    )

    # --------------------------------------------------------
    # 7. Save results
    # --------------------------------------------------------

    output = (
        MACRO / "27e_state_macro_asof_monthly.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    notes = [
        "Task 27.5C: Lagged macro as-of proxy.",
        "",
        "NOT archived real-time historical vintages.",
        "",
        "BLS availability assumption:",
        "Observed month-end + 1 calendar month.",
        "",
        "FHFA HPI availability assumption:",
        "Quarter-end + 3 calendar months.",
        "",
        "PMMS assumption:",
        "Weekly observation date treated as publication date.",
        "",
        "Official missing BLS observations are not imputed.",
        "Last previously available valid observations may be used.",
        "Their actual observation dates and ages are retained.",
        "",
        "Historical series may include subsequent revisions.",
        "PMMS historical methodology was revised in 2022.",
        "",
        "No future 36-month macro scenario is generated.",
    ]

    (
        REPORT / "27e_macro_asof_assumptions.txt"
    ).write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 8. Concise terminal output
    # --------------------------------------------------------

    print("\n=== AS-OF MACRO QA ===")

    print(
        f"States: {len(states)} | "
        f"Months: {len(dates)} | "
        f"Rows: {len(result):,}"
    )

    for col in [
        "hpi_state_sa",
        "hpi_yoy_pct",
        "unemployment_pct",
        "mortgage30_pct",
    ]:

        missing = result[col].isna().sum()
        pct = result[col].isna().mean() * 100

        print(
            f"{col}: missing {missing:,} ({pct:.2f}%)"
        )

    print("\n=== EXAMPLE: CALIFORNIA ===")

    selected = result[
        (result["state"] == "CA")
        & result["asof_yyyymm"].isin([
            200810, 202004, 202511
        ])
    ]

    columns = [
        "asof_yyyymm",
        "hpi_observation_end",
        "bls_observation_end",
        "pmms_observation_date",
        "hpi_state_sa",
        "unemployment_pct",
        "mortgage30_pct",
    ]

    print(
        selected[columns].to_string(index=False)
    )

    print(f"\nSaved: {output}")

    print(
        "WARNING: These are lagged historical proxies, "
        "not true archived point-in-time vintages."
    )


if __name__ == "__main__":
    main()
