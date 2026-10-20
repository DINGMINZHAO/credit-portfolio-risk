
from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path.home() / "Desktop" / "credit-portfolio-data"
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def from_ym(value):
    try:
        x = int(str(value).strip())
        y, m = divmod(x, 100)
        if 1900 <= y <= 2100 and 1 <= m <= 12:
            return y * 12 + m - 1
    except (ValueError, TypeError):
        pass
    return None


def as_code(value):
    value = str(value).strip()
    return value.zfill(2) if value.isdigit() else value


need = [
    "loan_id", "vintage", "admitted_60", "t0_m",
    "exit_m", "last_m", "exit_code", "known_60",
    "exit_within_36", "exit_within_60",
    "status_36", "status_60"
]

df = pd.read_parquet(
    DATA / "13_default_to_loss_followup.parquet",
    columns=need
)

d = df.loc[df["admitted_60"].eq(True)].copy()
d["loan_id"] = d["loan_id"].astype(str).str.strip()

if d["loan_id"].duplicated().any():
    raise ValueError("Duplicate loans in cohort")

# One bit per month after first 90+, months 1 to 60.
track = {
    lid: {
        "mask": 0,
        "last": None,
        "code": "",
        "effective": "",
        "rows": 0,
        "duplicate": 0,
    }
    for lid in d["loan_id"]
}

origin = dict(zip(
    d["loan_id"], d["t0_m"].astype(int)
))

print(
    f"Auditing {len(d):,} loans "
    "against six raw Performance ZIPs..."
)

for year in range(2006, 2012):
    path = RAW / f"sample_{year}.zip"

    with zipfile.ZipFile(path) as archive:
        matches = [
            x for x in archive.namelist()
            if x.endswith(
                f"sample_perf_{year}.txt"
            )
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Performance file not uniquely "
                f"found in {path}"
            )

        with archive.open(matches[0]) as f:
            for line in f:
                fields = (
                    line.decode(
                        "utf-8", errors="replace"
                    )
                    .rstrip("\r\n")
                    .split("|", 11)
                )

                if len(fields) < 10:
                    continue

                lid = fields[0].strip()

                if lid not in track:
                    continue

                month = from_ym(fields[1])

                if month is None:
                    continue

                s = track[lid]
                s["rows"] += 1

                dt = month - origin[lid]

                if 1 <= dt <= 60:
                    bit = 1 << (dt - 1)

                    if s["mask"] & bit:
                        s["duplicate"] += 1

                    s["mask"] |= bit

                if (
                    s["last"] is None
                    or month >= s["last"]
                ):
                    s["last"] = month
                    s["code"] = as_code(fields[8])
                    s["effective"] = fields[9].strip()

    print(f"  Read {year}")


records = []

for r in d.itertuples(index=False):
    s = track[r.loan_id]

    exit_m = (
        None
        if pd.isna(r.exit_m)
        else int(r.exit_m)
    )

    last_m = (
        None
        if pd.isna(r.last_m)
        else int(r.last_m)
    )

    t0 = int(r.t0_m)

    # Expect monthly records before exit,
    # but never demand records after exit.
    eligible_months = (
        60
        if exit_m is None
        else min(60, exit_m - t0 - 1)
    )

    eligible_months = max(
        0, eligible_months
    )

    expected = (1 << eligible_months) - 1

    missing_n = (
        expected & ~s["mask"]
    ).bit_count()

    after_last = (
        exit_m is not None
        and last_m is not None
        and exit_m > last_m
    )

    miss36 = (
        not bool(r.exit_within_36)
        and pd.isna(r.status_36)
    )

    miss60 = (
        not bool(r.exit_within_60)
        and pd.isna(r.status_60)
    )

    raw_exit = from_ym(s["effective"])

    records.append({
        "loan_id": r.loan_id,
        "vintage": int(r.vintage),
        "exit_code": r.exit_code,
        "t0_month_index": t0,
        "exit_month_index": exit_m,
        "parquet_last_month": last_m,
        "raw_last_month": s["last"],
        "raw_last_exit_code": s["code"],
        "raw_last_exit_month": raw_exit,
        "raw_rows": s["rows"],
        "duplicate_post90_months": s["duplicate"],
        "months_expected_before_exit_through_60": (
            eligible_months
        ),
        "missing_preexit_months": missing_n,
        "gap_exit_after_last": (
            exit_m - last_m
            if after_last else 0
        ),
        "exit_after_last": after_last,
        "missing_36m_status_no_exit": miss36,
        "missing_60m_status_no_exit": miss60,
        "raw_last_matches_parquet_last": (
            s["last"] == last_m
        ),
        "raw_last_code_matches_exit_code": (
            s["code"] == str(r.exit_code)
            if exit_m is not None else pd.NA
        ),
        "raw_last_effective_matches_exit_date": (
            raw_exit == exit_m
            if exit_m is not None else pd.NA
        ),
        "raw_has_36m_row": bool(
            s["mask"] & (1 << 35)
        ),
        "raw_has_60m_row": bool(
            s["mask"] & (1 << 59)
        ),
    })


a = pd.DataFrame(records)

ex = a.loc[
    a["exit_after_last"]
    | a["missing_36m_status_no_exit"]
    | a["missing_60m_status_no_exit"]
].copy()

ex.to_csv(
    OUT / "15_temporal_exceptions.csv",
    index=False
)

rows = []

groups = [("ALL", a)] + [
    (str(v), g)
    for v, g in a.groupby("vintage")
]

for vintage, g in groups:
    rows.append({
        "vintage": vintage,
        "loans": len(g),
        "loans_with_missing_preexit_months": int(
            (
                g["missing_preexit_months"] > 0
            ).sum()
        ),
        "total_missing_preexit_months": int(
            g["missing_preexit_months"].sum()
        ),
        "p95_missing_preexit_months": float(
            g["missing_preexit_months"]
            .quantile(.95)
        ),
        "exit_after_last": int(
            g["exit_after_last"].sum()
        ),
        "last_month_mismatch_raw_parquet": int(
            (
                ~g["raw_last_matches_parquet_last"]
            ).sum()
        ),
        "no_exit_36m_missing_status": int(
            g["missing_36m_status_no_exit"].sum()
        ),
        "no_exit_60m_missing_status": int(
            g["missing_60m_status_no_exit"].sum()
        ),
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "15_monthly_coverage_audit.csv",
    index=False
)

print("\n=== MONTHLY COVERAGE ===")
print(summary.to_string(index=False))

print(
    "\n=== EXIT AFTER LAST REPORT: "
    "MONTH GAPS ==="
)

print(
    a.loc[
        a["exit_after_last"],
        "gap_exit_after_last"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\n=== EXIT AFTER LAST REPORT: BY CODE ==="
)

print(
    a.loc[
        a["exit_after_last"],
        "exit_code"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n=== EXCEPTION RAW VALIDATION ===")

print("flagged loans:", len(ex))

print(
    "raw final period differs:",
    int((
        ~ex["raw_last_matches_parquet_last"]
    ).sum())
)

x = ex.loc[ex["exit_after_last"]]

print(
    "raw exit code mismatches:",
    int((
        x["raw_last_code_matches_exit_code"]
        == False
    ).sum())
)

print(
    "raw exit date mismatches:",
    int((
        x["raw_last_effective_matches_exit_date"]
        == False
    ).sum())
)

print(
    "\nSaved 15_monthly_coverage_audit.csv "
    "and 15_temporal_exceptions.csv"
)
