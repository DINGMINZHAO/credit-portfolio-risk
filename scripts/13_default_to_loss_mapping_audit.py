
"""Task 12: Default-to-Loss Mapping & Resolution Horizon Audit.

Freddie Mac Release 47, 2006-2011 samples.

Important:
Actual Loss is the final observed snapshot, not a dated
loss cash flow or the initial disclosure date.
"""

from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path.home() / "Desktop" / "credit-portfolio-data"
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)

CUTOFF_YM = 202603
HORIZONS = (36, 60)

LOSS_CODES = {"02", "03", "09", "15"}
CREDIT_CODES = {"02", "03", "09"}


def read_cols(file, required, optional=()):
    if not file.exists():
        raise FileNotFoundError(f"Input not found: {file}")

    names = set(pq.read_schema(file).names)
    absent = sorted(set(required) - names)

    if absent:
        raise ValueError(
            f"Missing columns in {file.name}: {absent}\n"
            f"Available: {sorted(names)}"
        )

    columns = list(dict.fromkeys(
        list(required) + [c for c in optional if c in names]
    ))

    return pd.read_parquet(file, columns=columns)


def to_serial(s):
    """Convert YYYYMM to an integer month index."""
    v = pd.to_numeric(s, errors="coerce")
    yr = np.floor(v / 100)
    mo = v % 100

    ok = (
        v.notna()
        & (v == np.floor(v))
        & yr.between(1900, 2100)
        & mo.between(1, 12)
    )

    res = pd.Series(pd.NA, index=s.index, dtype="Int64")
    res.loc[ok] = (
        yr.loc[ok] * 12 + mo.loc[ok] - 1
    ).astype("int64")

    return res


def code2(value):
    if pd.isna(value) or str(value).strip() in (
        "", "nan", "None"
    ):
        return "NO_CODE"

    v = str(value).strip()

    try:
        return f"{int(float(v)):02d}"
    except ValueError:
        return v


def class_dpd(value):
    value = value.strip()

    if value == "RA":
        return "REO_acquisition"

    if not value.isdigit():
        return "unavailable"

    n = int(value)

    if n == 0:
        return "current"
    if n == 1:
        return "30_59"
    if n == 2:
        return "60_89"

    return "90_plus"


def scan_monthly_status(events):
    """Stream raw performance records without loading them all.

    Track:
    - First return to current status
    - First three consecutive current months
    - Status at 36M and 60M
    """

    # State layout:
    # 0 first current month
    # 1 first month completing 3 consecutive current months
    # 2 consecutive current-month count
    # 3 previous observed month
    # 4 status at 36M
    # 5 status at 60M

    state = {
        str(k): [None, None, 0, None, None, None]
        for k in events["loan_id"]
    }

    origin = dict(zip(
        events["loan_id"].astype(str),
        events["t0_m"].astype(int)
    ))

    exits = dict(zip(
        events["loan_id"].astype(str),
        events["exit_m"]
    ))

    for vintage in range(2006, 2012):
        zpath = RAW / f"sample_{vintage}.zip"

        if not zpath.exists():
            raise FileNotFoundError(
                f"Missing monthly ZIP: {zpath}"
            )

        ids = set(
            events.loc[
                events["vintage"] == vintage, "loan_id"
            ].astype(str)
        )

        with zipfile.ZipFile(zpath) as z:
            matches = [
                n for n in z.namelist()
                if n.endswith(f"sample_perf_{vintage}.txt")
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one performance TXT in "
                    f"{zpath.name}; found {matches}"
                )

            with z.open(matches[0]) as raw:
                reader = io.TextIOWrapper(
                    raw,
                    encoding="utf-8",
                    errors="replace"
                )

                for line in reader:
                    loan, sep, _ = line.partition("|")
                    loan = loan.strip()

                    if not sep or loan not in ids:
                        continue

                    parts = line.split("|", 5)

                    if len(parts) < 4:
                        continue

                    try:
                        ym = int(parts[1].strip())
                        mo = (
                            (ym // 100) * 12
                            + ym % 100 - 1
                        )
                    except ValueError:
                        continue

                    t0 = origin[loan]
                    dt = mo - t0

                    if dt < 1 or dt > 60:
                        continue

                    ex = exits[loan]

                    if not pd.isna(ex) and mo > int(ex):
                        continue

                    s = state[loan]
                    category = class_dpd(parts[3])

                    if category == "current":
                        if s[0] is None:
                            s[0] = mo

                        s[2] = (
                            s[2] + 1
                            if s[3] == mo - 1
                            else 1
                        )

                        if s[2] >= 3 and s[1] is None:
                            s[1] = mo
                    else:
                        s[2] = 0

                    s[3] = mo

                    if dt == 36:
                        s[4] = category

                    if dt == 60:
                        s[5] = category

        print(
            f"  Monthly status scanned: {vintage} "
            f"({len(ids):,} first-90 loans)"
        )

    return state


def summary_row(group, horizon, cohort_name):
    n = len(group)

    known = group[f"known_{horizon}"]
    exited = group[f"exit_within_{horizon}"]
    loss = group[f"loss_within_{horizon}"]

    positive = loss.eq("positive")

    applicable = (
        exited & group["exit_code"].isin(LOSS_CODES)
    )

    exitmonths = group.loc[
        exited, "months_to_exit"
    ]

    return {
        "cohort": cohort_name,
        "horizon_m": horizon,
        "vintage": (
            str(group["vintage"].iloc[0])
            if n else "ALL"
        ),
        "n": n,
        "known_outcome": int(known.sum()),
        "unknown_followup": int((~known).sum()),

        "exit_01": int((
            exited & group["exit_code"].eq("01")
        ).sum()),

        "credit_exit_02_03_09": int((
            exited & group["exit_code"].isin(CREDIT_CODES)
        ).sum()),

        "loan_sale_15": int((
            exited & group["exit_code"].eq("15")
        ).sum()),

        "exit_16": int((
            exited & group["exit_code"].eq("16")
        ).sum()),

        "exit_96": int((
            exited & group["exit_code"].eq("96")
        ).sum()),

        "other_exit": int((
            exited
            & ~group["exit_code"].isin(
                LOSS_CODES | {"01", "16", "96"}
            )
        ).sum()),

        "still_observed_no_exit": int((
            known & ~exited
        ).sum()),

        "applicable_exit": int(applicable.sum()),

        "reported_positive": int(positive.sum()),
        "reported_zero": int(
            loss.eq("zero").sum()
        ),
        "reported_negative": int(
            loss.eq("negative").sum()
        ),
        "applicable_missing": int(
            loss.eq("missing").sum()
        ),

        "missing_with_defect": int((
            loss.eq("missing") & group["has_defect"]
        ).sum()),

        "ever_current": int(
            group[f"ever_current_{horizon}"].sum()
        ),

        "three_current_consecutive": int(
            group[f"three_current_{horizon}"].sum()
        ),

        "no_exit_current_at_horizon": int((
            known
            & ~exited
            & group[f"status_{horizon}"].eq("current")
        ).sum()),

        "no_exit_90plus_at_horizon": int((
            known
            & ~exited
            & group[f"status_{horizon}"].eq("90_plus")
        ).sum()),

        "no_exit_missing_status_at_horizon": int((
            known
            & ~exited
            & group[f"status_{horizon}"].isna()
        ).sum()),

        "exit_delay_median_conditional": (
            round(float(exitmonths.median()), 2)
            if len(exitmonths) else np.nan
        ),

        "exit_delay_p90_conditional": (
            round(float(exitmonths.quantile(0.90)), 2)
            if len(exitmonths) else np.nan
        ),

        "observed_positive_share_pct_of_all": (
            round(float(100 * positive.sum() / n), 3)
            if n else np.nan
        ),

        "unknown_followup_pct": (
            round(float(100 * (~known).sum() / n), 3)
            if n else np.nan
        ),
    }


def main():
    source = read_cols(
        DATA / "11_historical_loss_2006_2011.parquet",
        [
            "loan_id",
            "vintage",
            "first90_recomputed",
            "exit_code",
            "exit_date",
            "defect_settlement",
            "actual_loss",
            "last_period",
        ],
        [
            "upb_at_first90",
            "zero_removal_upb",
        ],
    )

    if (
        len(source) != 300_000
        or source["loan_id"].duplicated().any()
    ):
        raise ValueError(
            "Historical loss table is not 300,000 "
            "unique loans; stop and inspect."
        )

    source["loan_id"] = (
        source["loan_id"].astype(str).str.strip()
    )

    source["vintage"] = pd.to_numeric(
        source["vintage"],
        errors="raise"
    ).astype(int)

    if set(source["vintage"].unique()) != set(
        range(2006, 2012)
    ):
        raise ValueError(
            "Unexpected vintage coverage in loss file"
        )

    # Independent event-date consistency check.
    events_ref = read_cols(
        DATA / "03_loan_events_2006_2022.parquet",
        [
            "loan_id",
            "vintage",
            "first_90_period",
        ],
    )

    events_ref = events_ref.loc[
        events_ref["vintage"].isin(range(2006, 2012))
    ].copy()

    if (
        len(events_ref) != 300_000
        or events_ref["loan_id"].duplicated().any()
    ):
        raise ValueError(
            "03 loan-event reference is not "
            "300,000 unique historical loans"
        )

    events_ref["loan_id"] = (
        events_ref["loan_id"].astype(str).str.strip()
    )

    source = source.merge(
        events_ref[["loan_id", "first_90_period"]],
        on="loan_id",
        how="left",
        validate="one_to_one",
    )

    source["t0_m"] = to_serial(
        source["first90_recomputed"]
    )

    source["ref_t0_m"] = to_serial(
        source["first_90_period"]
    )

    mismatch = (
        source["t0_m"].fillna(-1)
        != source["ref_t0_m"].fillna(-1)
    ).sum()

    if mismatch:
        raise ValueError(
            f"First 90 date disagrees with 03 events "
            f"for {mismatch} loans. Stop."
        )

    source["exit_code"] = (
        source["exit_code"].map(code2)
    )

    source["exit_m"] = to_serial(
        source["exit_date"]
    )

    source["last_m"] = to_serial(
        source["last_period"]
    )

    source["defect_m"] = to_serial(
        source["defect_settlement"]
    )

    source["has_defect"] = (
        source["defect_m"].notna()
    )

    source["actual_loss"] = pd.to_numeric(
        source["actual_loss"],
        errors="coerce"
    )

    source["months_to_exit"] = (
        source["exit_m"] - source["t0_m"]
    )

    cutoff_m = int(
        to_serial(pd.Series([CUTOFF_YM])).iloc[0]
    )

    has90 = source["t0_m"].notna()

    invalid = (
        (
            source["exit_code"].ne("NO_CODE")
            & source["exit_m"].isna()
        )
        | (
            source["exit_code"].eq("NO_CODE")
            & source["exit_m"].notna()
        )
        | (
            has90
            & source["exit_m"].notna()
            & (
                source["exit_m"]
                < source["t0_m"]
            )
        )
        | source["last_m"].isna()
        | (
            has90
            & (
                source["t0_m"]
                > source["last_m"]
            )
        )
        | (
            has90
            & (
                source["t0_m"]
                > cutoff_m
            )
        )
        | (
            source["exit_m"].notna()
            & (
                source["exit_m"]
                > cutoff_m
            )
        )
    ).fillna(False)

    source["valid_event_timeline"] = (
        has90 & ~invalid
    )

    quality = (
        source.assign(
            no_first90=~has90,
            invalid_timeline=invalid & has90
        )
        .groupby("vintage", as_index=False)
        .agg(
            all_loans=("loan_id", "size"),
            first90=("t0_m", "count"),
            invalid_timeline=(
                "invalid_timeline", "sum"
            ),
            no_first90=("no_first90", "sum"),
        )
    )

    quality.to_csv(
        OUT / "13_cohort_audit.csv",
        index=False
    )

    cohort = source.loc[
        source["valid_event_timeline"]
    ].copy()

    if cohort.empty:
        raise ValueError(
            "No valid first-90 cohort"
        )

    # Actual Loss status at the release snapshot.
    cohort["loss_status"] = "not_applicable"

    app = cohort["exit_code"].isin(LOSS_CODES)

    cohort.loc[
        app & cohort["actual_loss"].isna(),
        "loss_status"
    ] = "missing"

    cohort.loc[
        app & cohort["actual_loss"].gt(0),
        "loss_status"
    ] = "positive"

    cohort.loc[
        app & cohort["actual_loss"].eq(0),
        "loss_status"
    ] = "zero"

    cohort.loc[
        app & cohort["actual_loss"].lt(0),
        "loss_status"
    ] = "negative"

    # Cross-check Task 11 definitions if available.
    aux_path = (
        DATA / "12_lgd_definition_audit.parquet"
    )

    if aux_path.exists():
        aux = read_cols(
            aux_path,
            ["loan_id"],
            ["loss_status", "missing_reason"]
        )

        aux["loan_id"] = (
            aux["loan_id"].astype(str).str.strip()
        )

        if aux["loan_id"].duplicated().any():
            raise ValueError(
                "Duplicate loan IDs in 12 LGD audit"
            )

        cohort = cohort.merge(
            aux.rename(
                columns={
                    "loss_status": "loss_status_12"
                }
            ),
            on="loan_id",
            how="left",
            validate="one_to_one",
        )

        if "loss_status_12" in cohort:
            status_mismatch = (
                cohort["loss_status_12"].notna()
                & cohort["loss_status_12"].ne(
                    cohort["loss_status"]
                )
            )

            if status_mismatch.any():
                print(
                    "WARNING: "
                    f"{status_mismatch.sum()} "
                    "loss-status mismatches "
                    "with Task 11."
                )

    print(
        f"Full sample: {len(source):,}; "
        f"first 90+: {has90.sum():,}; "
        f"valid timelines: {len(cohort):,}"
    )

    print(
        "Scanning monthly outcomes "
        "(6 raw ZIPs; may take a few minutes)..."
    )

    states = scan_monthly_status(cohort)

    # Construct 36M and 60M follow-up outcomes.
    for h, ix in ((36, 4), (60, 5)):
        target = cohort["t0_m"] + h

        cohort[f"admitted_{h}"] = (
            target <= cutoff_m
        )

        cohort[f"exit_within_{h}"] = (
            cohort["exit_m"].notna()
            & cohort["exit_m"].between(
                cohort["t0_m"],
                target
            )
        )

        # Known exit within H; or verified survival
        # without formal exit through H.
        cohort[f"known_{h}"] = (
            cohort[f"exit_within_{h}"]
            | (
                cohort["exit_m"].notna()
                & (cohort["exit_m"] > target)
            )
            | (
                cohort["last_m"].notna()
                & (cohort["last_m"] >= target)
            )
        )

        cohort[f"loss_within_{h}"] = np.where(
            cohort[f"exit_within_{h}"],
            cohort["loss_status"],
            "not_exited_in_horizon"
        )

        cohort[f"status_{h}"] = (
            cohort["loan_id"].map(
                lambda k: states[k][ix]
            )
        )

        cohort[f"ever_current_{h}"] = (
            cohort.apply(
                lambda r: (
                    states[r["loan_id"]][0]
                    is not None
                    and (
                        states[r["loan_id"]][0]
                        - r["t0_m"]
                        <= h
                    )
                ),
                axis=1
            )
        )

        cohort[f"three_current_{h}"] = (
            cohort.apply(
                lambda r: (
                    states[r["loan_id"]][1]
                    is not None
                    and (
                        states[r["loan_id"]][1]
                        - r["t0_m"]
                        <= h
                    )
                ),
                axis=1
            )
        )

    rows = []
    breakdown = []

    comparisons = [
        ("horizon_specific", None),
        (
            "same_60m_cohort",
            cohort["admitted_60"]
        ),
    ]

    for group_name, eligible in comparisons:
        for h in HORIZONS:
            if eligible is None:
                selected = cohort.loc[
                    cohort[f"admitted_{h}"]
                ]
            else:
                selected = cohort.loc[eligible]

            grouped = [
                ("ALL", selected)
            ] + list(
                selected.groupby("vintage")
            )

            for vintage, g in grouped:
                summary = summary_row(
                    g,
                    h,
                    group_name
                )
                summary["vintage"] = str(vintage)
                rows.append(summary)

            exited = selected.loc[
                selected[f"exit_within_{h}"]
            ]

            if not exited.empty:
                b = (
                    exited.groupby(
                        [
                            "vintage",
                            "exit_code",
                            "loss_status"
                        ],
                        dropna=False
                    )
                    .size()
                    .rename("loans")
                    .reset_index()
                )

                b["cohort"] = group_name
                b["horizon_m"] = h
                breakdown.append(b)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUT / "13_horizon_summary.csv",
        index=False
    )

    if breakdown:
        pd.concat(
            breakdown,
            ignore_index=True
        ).to_csv(
            OUT / "13_exit_loss_breakdown.csv",
            index=False
        )

    cohort.drop(
        columns=["ref_t0_m"],
        errors="ignore"
    ).to_parquet(
        DATA / "13_default_to_loss_followup.parquet",
        index=False
    )

    cols = [
        "horizon_m",
        "n",
        "known_outcome",
        "unknown_followup",
        "credit_exit_02_03_09",
        "loan_sale_15",
        "reported_positive",
        "applicable_missing",
        "ever_current",
        "three_current_consecutive",
    ]

    print(
        "\nHorizon-specific cohort "
        "(ALL vintages):"
    )

    print(
        summary.loc[
            (
                summary["cohort"]
                == "horizon_specific"
            )
            & (
                summary["vintage"] == "ALL"
            ),
            cols
        ].to_string(index=False)
    )

    print(
        "\nSame 60M-admitted cohort "
        "(ALL vintages):"
    )

    print(
        summary.loc[
            (
                summary["cohort"]
                == "same_60m_cohort"
            )
            & (
                summary["vintage"] == "ALL"
            ),
            cols
        ].to_string(index=False)
    )

    print(
        "\nSaved: "
        "13_cohort_audit.csv, "
        "13_horizon_summary.csv, "
        "13_exit_loss_breakdown.csv"
    )

    print(
        "       data/processed/"
        "13_default_to_loss_followup.parquet"
    )

    print(
        "NOTE: Actual Loss is final observed "
        "net loss as of 2026-03 release cutoff,"
    )
    print(
        "      NOT an observed loss cash flow "
        "or initial reporting date."
    )


if __name__ == "__main__":
    main()
