
"""Task 27.5A — Macro Data Download (HTML Fallback)

Sources:
1. FHFA State Purchase-Only HPI (quarterly)
2. BLS State Unemployment via FRED (monthly)
3. Freddie Mac PMMS via FRED (weekly)

Period: 2005-01 through 2026-03.

Features:
- Reuses existing FHFA download
- Downloads 51 unemployment series
- Preserves official missing observations as NaN
- Supports caching and resuming
- Validates date coverage and values
- Does not modify loan-level data
- Does not create future macroeconomic forecasts
"""

from pathlib import Path
from urllib.request import Request, urlopen
from html.parser import HTMLParser
from datetime import datetime, timezone
from collections import Counter

import re
import time
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

ROOT = Path.cwd()

RAW = ROOT / "data/external/macro_raw"
PROC = ROOT / "data/processed/macro"
REPORT = ROOT / "outputs/pd_model"

START = pd.Timestamp("2005-01-01")
END = pd.Timestamp("2026-03-31")

FHFA_URL = (
    "https://www.fhfa.gov/hpi/download/"
    "quarterly_datasets/hpi_po_state.txt"
)

for folder in [RAW, PROC, REPORT]:
    folder.mkdir(parents=True, exist_ok=True)


STATES = [
    "AL", "AK", "AZ", "AR", "CA",
    "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI",
    "WY",
]

assert len(STATES) == 51


# ============================================================
# 1. HTML parser
# ============================================================

class PageText(HTMLParser):

    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if self.hidden == 0:
            self.parts.append(data)


# ============================================================
# 2. Parse FRED historical observations
# ============================================================

def parse_fred_html(html, series_id):

    parser = PageText()
    parser.feed(html)

    text = " ".join(parser.parts)

    # FRED may report an unavailable value as ".".
    #
    # Examples:
    # 2025-09-01 4.7
    # 2025-10-01 .
    # 2025-11-01 4.8

    pattern = (
        r"(?<!\d)"
        r"(\d{4}-\d{2}-\d{2})"
        r"\s+"
        r"(\.|-?(?:\d+(?:\.\d*)?|\.\d+))"
        r"(?=\s|$)"
    )

    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError(
            f"{series_id}: no historical observations parsed"
        )

    df = pd.DataFrame(
        matches,
        columns=["date", "value"],
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # "." automatically becomes NaN.
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    ).copy()

    df = df[
        df["date"].between(START, END)
    ].copy()

    return df.sort_values(
        "date"
    ).reset_index(drop=True)


# ============================================================
# 3. Validate FRED observations
# ============================================================

def validate_fred(df, series_id):

    if df.empty:
        raise ValueError(
            f"{series_id}: empty dataset"
        )

    if df["date"].duplicated().any():
        raise ValueError(
            f"{series_id}: duplicate observation dates"
        )

    valid_values = df["value"].dropna()

    if not valid_values.between(0, 30).all():
        raise ValueError(
            f"{series_id}: invalid rate values"
        )

    if series_id.endswith("UR"):

        expected = pd.date_range(
            "2005-01-01",
            "2026-03-01",
            freq="MS",
        )

        observed = pd.DatetimeIndex(
            df["date"]
        )

        # An absent date row is different from an
        # officially reported missing numeric value.
        missing_dates = expected.difference(
            observed
        )

        if len(missing_dates):
            missing_list = (
                missing_dates
                .strftime("%Y-%m")
                .tolist()
            )

            raise ValueError(
                f"{series_id}: date rows missing "
                f"from downloaded table: {missing_list}"
            )

        if len(df) != len(expected):
            raise ValueError(
                f"{series_id}: unexpected observation count"
            )

    elif series_id == "MORTGAGE30US":

        # Expected approximately 21 years of weekly data.
        if valid_values.size < 900:
            raise ValueError(
                "MORTGAGE30US: insufficient weekly observations"
            )

        if df["date"].min() > pd.Timestamp("2005-02-01"):
            raise ValueError(
                "MORTGAGE30US: early historical coverage missing"
            )

        if df["date"].max() < pd.Timestamp("2026-03-01"):
            raise ValueError(
                "MORTGAGE30US: recent historical coverage missing"
            )


# ============================================================
# 4. Download FRED series with caching
# ============================================================

def get_fred_series(series_id):

    cache = RAW / f"fred_{series_id}_html.csv"

    # Reuse previously completed downloads.
    if cache.exists() and cache.stat().st_size > 100:

        cached = pd.read_csv(
            cache,
            parse_dates=["date"],
        )

        # Validate cached files too.
        try:
            validate_fred(cached, series_id)
            return cached

        except ValueError:
            print(
                f"Refreshing invalid cache: {series_id}",
                flush=True,
            )
            cache.unlink()

    url = (
        f"https://fred.stlouisfed.org/data/{series_id}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,*/*",
        },
    )

    last_error = None

    for attempt in range(3):

        try:

            with urlopen(
                request,
                timeout=45,
            ) as response:

                html = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

            df = parse_fred_html(
                html,
                series_id,
            )

            validate_fred(
                df,
                series_id,
            )

            # Cache only successfully validated data.
            temp = cache.with_suffix(".partial")

            df.to_csv(
                temp,
                index=False,
            )

            temp.replace(cache)

            return df

        except Exception as exc:

            last_error = exc

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Failed downloading {series_id}\n"
        f"URL: {url}\n"
        f"Error: {last_error}"
    )


# ============================================================
# 5. Read cached FHFA house price index
# ============================================================

def prepare_hpi():

    path = RAW / "fhfa_hpi_po_state.txt"

    if not path.exists():

        raise FileNotFoundError(
            f"FHFA file not found:\n{path}\n"
            "The previous script should already "
            "have downloaded this file."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )

    df.columns = (
        df.columns.str.strip().str.lower()
    )

    required = {
        "state", "yr", "qtr",
        "indx_sa", "indx_nsa",
    }

    if not required.issubset(df.columns):

        raise ValueError(
            f"Unexpected FHFA columns: {list(df.columns)}"
        )

    df["state"] = (
        df["state"].str.strip().str.upper()
    )

    for col in [
        "yr", "qtr", "indx_sa", "indx_nsa"
    ]:

        df[col] = pd.to_numeric(
            df[col].str.strip(),
            errors="coerce",
        )

    df = df[
        df["yr"].between(2005, 2026)
        & df["qtr"].between(1, 4)
    ].copy()

    df["period_end"] = (
        pd.PeriodIndex.from_fields(
            year=df["yr"].astype(int),
            quarter=df["qtr"].astype(int),
            freq="Q",
        )
        .to_timestamp(how="end")
        .normalize()
    )

    df = df[
        df["period_end"] <= END
    ].copy()

    df = df.rename(
        columns={
            "indx_sa": "hpi_sa",
            "indx_nsa": "hpi_nsa",
        }
    )

    df = df[
        [
            "state",
            "yr",
            "qtr",
            "period_end",
            "hpi_sa",
            "hpi_nsa",
        ]
    ]

    if df.duplicated(
        ["state", "period_end"]
    ).any():

        raise ValueError(
            "Duplicate FHFA state-quarter"
        )

    if (
        df["hpi_sa"].isna().any()
        or (df["hpi_sa"] <= 0).any()
    ):

        raise ValueError(
            "Invalid FHFA HPI values"
        )

    missing_states = set(STATES) - set(df["state"])

    if missing_states:

        raise ValueError(
            f"Missing FHFA states: {missing_states}"
        )

    return df.sort_values(
        ["state", "period_end"]
    ).reset_index(drop=True)


# ============================================================
# 6. Download state unemployment
# ============================================================

def prepare_unemployment():

    print(
        "Downloading BLS unemployment via FRED...",
        flush=True,
    )

    frames = []
    missing_summary = Counter()

    for i, state in enumerate(STATES, start=1):

        series_id = f"{state}UR"

        df = get_fred_series(series_id)

        # Preserve every month, including NaN values.
        df["state"] = state

        df["period_end"] = (
            df["date"]
            .dt.to_period("M")
            .dt.to_timestamp("M")
        )

        missing = df.loc[
            df["value"].isna(),
            "date",
        ]

        for date in missing:
            missing_summary[
                date.strftime("%Y-%m")
            ] += 1

        df = df.rename(
            columns={
                "value": "unemployment_pct",
            }
        )

        frames.append(
            df[
                [
                    "state",
                    "period_end",
                    "unemployment_pct",
                ]
            ]
        )

        if (
            i % 10 == 0
            or i == len(STATES)
        ):

            print(
                f"States completed: {i}/{len(STATES)}",
                flush=True,
            )

        time.sleep(0.2)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.sort_values(
        ["state", "period_end"]
    ).reset_index(drop=True)

    if result["state"].nunique() != 51:

        raise ValueError(
            "Expected 50 states plus DC"
        )

    if result.duplicated(
        ["state", "period_end"]
    ).any():

        raise ValueError(
            "Duplicate state-month"
        )

    print("\n=== OFFICIAL MISSING OBSERVATIONS ===")

    if missing_summary:

        for month, count in sorted(
            missing_summary.items()
        ):

            print(
                f"{month}: {count} state(s) "
                "with missing unemployment values"
            )

    else:
        print("None")

    return result


# ============================================================
# 7. Download mortgage rates
# ============================================================

def prepare_mortgage_rates():

    print(
        "\nDownloading Freddie Mac PMMS via FRED...",
        flush=True,
    )

    df = get_fred_series(
        "MORTGAGE30US"
    )

    df = df.rename(
        columns={
            "value": "mortgage30_pct",
        }
    )

    return df[
        [
            "date",
            "mortgage30_pct",
        ]
    ].sort_values(
        "date"
    ).reset_index(drop=True)


# ============================================================
# 8. Coverage statistics
# ============================================================

def describe(name, df, date_col, geo_col=None):

    return {
        "source": name,
        "rows": len(df),
        "locations": (
            df[geo_col].nunique()
            if geo_col else 1
        ),
        "first": str(
            df[date_col].min().date()
        ),
        "last": str(
            df[date_col].max().date()
        ),
        "null_cells": int(
            df.isna().sum().sum()
        ),
    }


# ============================================================
# 9. Main
# ============================================================

def main():

    print(
        "Reading cached FHFA HPI...",
        flush=True,
    )

    hpi = prepare_hpi()

    unemployment = prepare_unemployment()

    rates = prepare_mortgage_rates()

    # --------------------------------------------------------
    # Save standardized datasets
    # --------------------------------------------------------

    hpi.to_csv(
        PROC / "27d_fhfa_state_hpi_quarterly.csv",
        index=False,
    )

    unemployment.to_csv(
        PROC / "27d_bls_state_unemployment_monthly.csv",
        index=False,
    )

    rates.to_csv(
        PROC / "27d_pmms_mortgage30_weekly.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Coverage report
    # --------------------------------------------------------

    coverage = pd.DataFrame([
        describe(
            "FHFA state HPI quarterly",
            hpi,
            "period_end",
            "state",
        ),

        describe(
            "BLS unemployment via FRED",
            unemployment,
            "period_end",
            "state",
        ),

        describe(
            "Freddie Mac PMMS weekly",
            rates,
            "date",
        ),
    ])

    coverage.to_csv(
        REPORT / "27d_macro_coverage.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Source manifest
    # --------------------------------------------------------

    manifest = [
        "Task 27.5A Macro Data Sources",
        "",
        "Download UTC:",
        datetime.now(
            timezone.utc
        ).isoformat(),
        "",
        f"Observation start: {START.date()}",
        f"Observation cutoff: {END.date()}",
        "",
        f"FHFA: {FHFA_URL}",
        "BLS unemployment:",
        "https://fred.stlouisfed.org/data/{STATE}UR",
        "Freddie Mac PMMS:",
        "https://fred.stlouisfed.org/data/MORTGAGE30US",
        "",
        "IMPORTANT:",
        "Official missing observations are preserved as NaN.",
        "No missing macro observations were imputed.",
        "These files may include historical data revisions.",
        "Observation dates are NOT publication dates.",
        "Historical point-in-time availability is NOT verified.",
        "Future actual macro data are NOT forecast scenarios.",
        "PMMS methodology changed in November 2022.",
        "No loan-level data were modified.",
    ]

    (
        REPORT / "27d_macro_source_manifest.txt"
    ).write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n=== MACRO COVERAGE ===")

    print(
        coverage.to_string(index=False)
    )

    print("\n=== SELECTED YEARS ===")

    for year in [
        2006, 2008, 2015,
        2020, 2023, 2025
    ]:

        h = hpi[
            hpi["period_end"].dt.year == year
        ]

        u = unemployment[
            unemployment["period_end"].dt.year == year
        ]

        r = rates[
            rates["date"].dt.year == year
        ]

        print(
            f"{year}: "
            f"HPI {len(h)} state-quarters | "
            f"BLS {len(u)} state-months | "
            f"PMMS {len(r)} weeks"
        )

    print("\n=== MISSING VALUE SUMMARY ===")

    print(
        "HPI missing:",
        int(hpi.isna().sum().sum()),
    )

    print(
        "Unemployment missing:",
        int(
            unemployment[
                "unemployment_pct"
            ].isna().sum()
        ),
    )

    print(
        "Mortgage rate missing:",
        int(
            rates[
                "mortgage30_pct"
            ].isna().sum()
        ),
    )

    print(
        "\nSaved 3 standardized macro datasets "
        "and 2 audit reports."
    )


if __name__ == "__main__":
    main()
