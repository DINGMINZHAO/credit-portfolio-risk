
"""Task 27.5A — Resume macro download using FRED for BLS data."""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
import importlib.util
import time

import pandas as pd


ROOT = Path.cwd()
RAW = ROOT / "data/external/macro_raw"
PROC = ROOT / "data/processed/macro"
REPORT = ROOT / "outputs/pd_model"

START = pd.Timestamp("2005-01-01")
CUTOFF = pd.Timestamp("2026-03-31")

for folder in [RAW, PROC, REPORT]:
    folder.mkdir(parents=True, exist_ok=True)

# Reuse tested parsing functions from the previous script.
old_script = ROOT / "scripts/27_download_macro_data.py"

spec = importlib.util.spec_from_file_location(
    "previous_macro", old_script
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

STATES = sorted(set(old.FIPS_TO_STATE.values()))


def fred_download(series_id):
    """Download a FRED CSV; cache successful downloads."""

    path = RAW / f"fred_{series_id}.csv"

    if path.exists() and path.stat().st_size > 100:
        return path

    url = (
        "https://fred.stlouisfed.org/graph/"
        f"fredgraph.csv?id={series_id}"
    )

    last_error = None

    for attempt in range(3):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/csv,*/*",
                },
            )

            with urlopen(request, timeout=45) as response:
                content = response.read()

            if len(content) < 100:
                raise ValueError("Unexpectedly small CSV")

            temp = path.with_suffix(".partial")
            temp.write_bytes(content)
            temp.replace(path)

            return path

        except (HTTPError, URLError, OSError, ValueError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"FRED download failed: {series_id}\n"
        f"URL: {url}\nError: {last_error}"
    )


def prepare_state_unemployment():

    frames = []

    for i, state in enumerate(STATES, start=1):

        series = f"{state}UR"
        path = fred_download(series)

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        date_col = (
            "observation_date"
            if "observation_date" in df.columns
            else "DATE"
        )

        if date_col not in df or series not in df:
            raise ValueError(
                f"Unexpected FRED schema for {series}: "
                f"{list(df.columns)}"
            )

        dates = pd.to_datetime(
            df[date_col], errors="coerce"
        )

        values = pd.to_numeric(
            df[series], errors="coerce"
        )

        part = pd.DataFrame({
            "state": state,
            "period_end": (
                dates.dt.to_period("M")
                .dt.to_timestamp("M")
            ),
            "unemployment_pct": values,
        })

        part = part[
            part["period_end"].between(START, CUTOFF)
        ].dropna().copy()

        if part.empty:
            raise ValueError(f"No observations: {series}")

        if part["period_end"].duplicated().any():
            raise ValueError(f"Duplicate months: {series}")

        if not part["unemployment_pct"].between(0, 100).all():
            raise ValueError(f"Invalid unemployment values: {series}")

        expected_months = pd.date_range(
            START, CUTOFF, freq="ME"
        )

        missing = expected_months.difference(
            pd.DatetimeIndex(part["period_end"])
        )

        if len(missing):
            raise ValueError(
                f"{series}: {len(missing)} missing months; "
                f"first missing: {missing[0]}"
            )

        frames.append(part)

        if i % 10 == 0 or i == len(STATES):
            print(
                f"FRED state unemployment: "
                f"{i}/{len(STATES)} completed",
                flush=True
            )

        time.sleep(0.15)

    result = pd.concat(
        frames, ignore_index=True
    ).sort_values(
        ["state", "period_end"]
    ).reset_index(drop=True)

    if result["state"].nunique() != 51:
        raise ValueError("Expected 50 states plus DC")

    return result


def main():

    # The previous run already downloaded this file.
    hpi_path = RAW / "fhfa_hpi_po_state.txt"

    if not hpi_path.exists():
        hpi_path = old.download(
            "fhfa_hpi_po_state.txt",
            old.SOURCES["fhfa_hpi_po_state.txt"],
        )

    print("Reading cached FHFA HPI...", flush=True)

    hpi = old.prepare_hpi(hpi_path)

    print("Downloading BLS unemployment via FRED...", flush=True)

    unemployment = prepare_state_unemployment()

    print("Downloading Freddie Mac PMMS via FRED...", flush=True)

    rate_path = fred_download("MORTGAGE30US")

    rates = old.prepare_mortgage_rate(rate_path)

    # Preserve the original output interface.
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

    coverage = pd.DataFrame([
        old.describe(
            "FHFA state HPI quarterly",
            hpi, "period_end", "state"
        ),
        old.describe(
            "BLS unemployment via FRED",
            unemployment, "period_end", "state"
        ),
        old.describe(
            "Freddie Mac PMMS weekly",
            rates, "date"
        ),
    ])

    coverage.to_csv(
        REPORT / "27d_macro_coverage.csv",
        index=False,
    )

    manifest = [
        f"Download UTC: {datetime.now(timezone.utc).isoformat()}",
        "Observation cutoff: 2026-03-31",
        "FHFA: https://www.fhfa.gov/hpi/",
        "BLS state unemployment: FRED {STATE}UR series",
        "PMMS: FRED MORTGAGE30US",
        "",
        "IMPORTANT:",
        "All historical data may contain later revisions.",
        "Observation dates are not publication dates.",
        "These are not historical point-in-time vintages.",
        "Future observed macro data are not forecast scenarios.",
    ]

    (
        REPORT / "27d_macro_source_manifest.txt"
    ).write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )

    print("\n=== MACRO COVERAGE ===")
    print(coverage.to_string(index=False))

    print("\n=== SELECTED YEARS ===")

    for year in [2006, 2008, 2015, 2020, 2023, 2025]:

        a = hpi[hpi.period_end.dt.year == year]
        b = unemployment[
            unemployment.period_end.dt.year == year
        ]
        c = rates[rates.date.dt.year == year]

        print(
            f"{year}: "
            f"HPI {len(a)} state-quarters | "
            f"BLS {len(b)} state-months | "
            f"PMMS {len(c)} weeks"
        )

    print("\nSaved all 3 macro datasets.")


if __name__ == "__main__":
    main()
