
"""Task 27.5A — Download and audit mortgage macroeconomic data.

Sources:
FHFA state quarterly purchase-only HPI
BLS state monthly seasonally adjusted unemployment
Freddie Mac 30Y PMMS via FRED

No loan data are modified. No future scenarios are constructed.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
import shutil

import pandas as pd


ROOT = Path.cwd()
RAW = ROOT / "data/external/macro_raw"
PROC = ROOT / "data/processed/macro"
REPORT = ROOT / "outputs/pd_model"

CUTOFF = pd.Timestamp("2026-03-31")
START = pd.Timestamp("2005-01-01")

SOURCES = {
    "fhfa_hpi_po_state.txt":
        "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_po_state.txt",

    "bls_la.series":
        "https://download.bls.gov/pub/time.series/la/la.series",

    "bls_la.data.3.AllStatesS":
        "https://download.bls.gov/pub/time.series/la/la.data.3.AllStatesS",

    "fred_MORTGAGE30US.csv":
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
}

FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR",
    "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX",
    "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


# ==========================================================
# 1. Download with caching
# ==========================================================

def download(name, url):

    path = RAW / name

    if path.exists() and path.stat().st_size >= 100:
        print(f"CACHED {name}", flush=True)
        return path

    temp = path.with_name(path.name + ".partial")

    try:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,text/plain,*/*",
            },
        )

        with urlopen(request, timeout=90) as response:
            with temp.open("wb") as out:
                shutil.copyfileobj(response, out)

        if temp.stat().st_size < 100:
            raise ValueError("Empty or truncated download")

        temp.replace(path)

    except (HTTPError, URLError, OSError, ValueError) as exc:

        temp.unlink(missing_ok=True)

        raise RuntimeError(
            f"Download failed: {url}\n"
            f"Error: {exc}\n"
            f"If automated access is blocked, download the "
            f"file manually and save it as:\n{path}"
        ) from exc

    print(
        f"DOWNLOADED {name} "
        f"({path.stat().st_size / 1048576:.1f} MB)",
        flush=True,
    )

    return path


# ==========================================================
# 2. FHFA state quarterly house-price index
# ==========================================================

def prepare_hpi(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )

    df.columns = df.columns.str.strip().str.lower()

    required = {
        "state", "yr", "qtr", "indx_sa", "indx_nsa"
    }

    if not required <= set(df.columns):
        raise ValueError(
            f"Unexpected FHFA columns: {list(df.columns)}"
        )

    df["state"] = (
        df["state"].str.strip().str.upper()
    )

    for col in ["yr", "qtr", "indx_sa", "indx_nsa"]:
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
        df["period_end"] <= CUTOFF
    ].copy()

    df = df.rename(columns={
        "indx_sa": "hpi_sa",
        "indx_nsa": "hpi_nsa",
    })

    df = df[
        [
            "state", "yr", "qtr",
            "period_end", "hpi_sa", "hpi_nsa",
        ]
    ]

    if df.duplicated(["state", "period_end"]).any():
        raise ValueError("Duplicate FHFA state-quarter")

    if (
        df["hpi_sa"].isna().any()
        or (df["hpi_sa"] <= 0).any()
    ):
        raise ValueError("Invalid FHFA HPI values")

    expected = set(FIPS_TO_STATE.values())
    missing_states = expected - set(df["state"])

    if missing_states:
        raise ValueError(
            f"Missing FHFA states: {missing_states}"
        )

    return df.sort_values(
        ["state", "period_end"]
    ).reset_index(drop=True)


# ==========================================================
# 3. BLS state monthly unemployment
# ==========================================================

def prepare_unemployment(metadata_path, data_path):

    meta = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    meta.columns = meta.columns.str.strip()

    for col in meta.columns:
        meta[col] = meta[col].str.strip()

    required = {
        "series_id", "area_type_code",
        "area_code", "measure_code", "seasonal",
    }

    if not required <= set(meta.columns):
        raise ValueError("Unexpected BLS metadata schema")

    # State, unemployment rate, seasonally adjusted.
    selected = meta[
        (meta["area_type_code"] == "A")
        & (meta["measure_code"] == "03")
        & (meta["seasonal"] == "S")
        & meta["area_code"].str.fullmatch(
            r"ST\d{2}0{11}", na=False
        )
    ].copy()

    selected["state"] = (
        selected["area_code"]
        .str[2:4]
        .map(FIPS_TO_STATE)
    )

    selected = selected.dropna(subset=["state"])

    if (
        selected["state"].nunique() != 51
        or selected["state"].duplicated().any()
    ):
        raise ValueError(
            "Expected 50 states plus DC in BLS metadata"
        )

    series_lookup = selected.set_index(
        "series_id"
    )["state"]

    wanted = set(series_lookup.index)
    pieces = []

    # Stream the source file; retain only the 51 target series.
    reader = pd.read_csv(
        data_path,
        sep="\t",
        dtype=str,
        chunksize=150_000,
    )

    for part in reader:

        part.columns = part.columns.str.strip()

        for col in [
            "series_id", "year", "period", "value"
        ]:
            part[col] = part[col].str.strip()

        mask = (
            part["series_id"].isin(wanted)
            & part["period"].str.fullmatch(
                r"M(0[1-9]|1[0-2])",
                na=False,
            )
        )

        sub = part.loc[
            mask,
            ["series_id", "year", "period", "value"],
        ]

        if not sub.empty:
            pieces.append(sub.copy())

    if not pieces:
        raise ValueError("No BLS state unemployment data")

    df = pd.concat(
        pieces, ignore_index=True
    )

    df["state"] = (
        df["series_id"].map(series_lookup)
    )

    df["year"] = pd.to_numeric(
        df["year"], errors="coerce"
    )

    df["month"] = pd.to_numeric(
        df["period"].str[1:],
        errors="coerce",
    )

    df["unemployment_pct"] = pd.to_numeric(
        df["value"], errors="coerce"
    )

    df = df[
        df["year"].between(2005, 2026)
    ].copy()

    df["period_end"] = (
        pd.to_datetime(
            dict(
                year=df["year"].astype(int),
                month=df["month"].astype(int),
                day=1,
            )
        )
        + pd.offsets.MonthEnd(0)
    )

    df = df[
        df["period_end"] <= CUTOFF
    ].copy()

    df = df[
        ["state", "period_end", "unemployment_pct"]
    ].sort_values(
        ["state", "period_end"]
    )

    if df.duplicated(
        ["state", "period_end"]
    ).any():
        raise ValueError("Duplicate BLS state-month")

    if (
        df["unemployment_pct"].isna().any()
        or not df["unemployment_pct"]
        .between(0, 100).all()
    ):
        raise ValueError("Invalid BLS unemployment values")

    return df.reset_index(drop=True)


# ==========================================================
# 4. Freddie Mac weekly mortgage rate via FRED
# ==========================================================

def prepare_mortgage_rate(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    date_col = (
        "observation_date"
        if "observation_date" in df.columns
        else "DATE"
    )

    if (
        date_col not in df.columns
        or "MORTGAGE30US" not in df.columns
    ):
        raise ValueError(
            f"Unexpected PMMS schema: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df["mortgage30_pct"] = pd.to_numeric(
        df["MORTGAGE30US"],
        errors="coerce",
    )

    df = df[
        df["date"].between(START, CUTOFF)
    ].dropna(
        subset=["date", "mortgage30_pct"]
    ).copy()

    if df["date"].duplicated().any():
        raise ValueError("Duplicate PMMS observation date")

    if not df["mortgage30_pct"].between(0, 30).all():
        raise ValueError("Invalid mortgage rates")

    return df[
        ["date", "mortgage30_pct"]
    ].sort_values("date").reset_index(drop=True)


# ==========================================================
# 5. Coverage summary
# ==========================================================

def describe(name, frame, date_col, geo_col=None):

    if frame.empty:
        raise ValueError(f"No observations: {name}")

    return {
        "source": name,
        "rows": len(frame),
        "locations": (
            frame[geo_col].nunique()
            if geo_col else 1
        ),
        "first": str(frame[date_col].min().date()),
        "last": str(frame[date_col].max().date()),
        "null_cells": int(
            frame.isna().sum().sum()
        ),
    }


# ==========================================================
# 6. Main
# ==========================================================

def main():

    for folder in [RAW, PROC, REPORT]:
        folder.mkdir(parents=True, exist_ok=True)

    paths = {
        name: download(name, url)
        for name, url in SOURCES.items()
    }

    hpi = prepare_hpi(
        paths["fhfa_hpi_po_state.txt"]
    )

    unemployment = prepare_unemployment(
        paths["bls_la.series"],
        paths["bls_la.data.3.AllStatesS"],
    )

    rates = prepare_mortgage_rate(
        paths["fred_MORTGAGE30US.csv"]
    )

    # Standardized source tables.
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
        describe(
            "FHFA state HPI quarterly",
            hpi, "period_end", "state",
        ),
        describe(
            "BLS state unemployment monthly",
            unemployment, "period_end", "state",
        ),
        describe(
            "Freddie Mac PMMS weekly",
            rates, "date",
        ),
    ])

    coverage.to_csv(
        REPORT / "27d_macro_coverage.csv",
        index=False,
    )

    # Source provenance and important modelling warnings.
    with (
        REPORT / "27d_macro_source_manifest.txt"
    ).open("w", encoding="utf-8") as f:

        f.write(
            "Downloaded at UTC: "
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )

        f.write(
            f"Observation cutoff: {CUTOFF.date()}\n"
        )

        for name, url in SOURCES.items():
            f.write(f"{name}: {url}\n")

        f.write(
            "\nIMPORTANT:\n"
            "Current files may contain historical revisions.\n"
            "Observation period is not publication date.\n"
            "This output is NOT a real-time vintage dataset.\n"
            "Future observed macro data must not be used as "
            "future forecasts.\n"
            "No loan-level data have been modified.\n"
        )

    print("\n=== MACRO COVERAGE ===")
    print(coverage.to_string(index=False))

    print("\n=== SELECTED YEARS ===")

    for year in [2006, 2008, 2015, 2020, 2023, 2025]:

        a = hpi[
            hpi["period_end"].dt.year == year
        ]

        b = unemployment[
            unemployment["period_end"].dt.year == year
        ]

        c = rates[
            rates["date"].dt.year == year
        ]

        print(
            f"{year}: "
            f"HPI {len(a)} state-quarters | "
            f"BLS {len(b)} state-months | "
            f"PMMS {len(c)} weeks"
        )

    print(
        "\nSaved 3 standardized macro datasets "
        "and 2 audit reports."
    )


if __name__ == "__main__":
    main()
