from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

PD_FILE = (
    ROOT / "outputs/modeling/task32_12m"
    / "12m_lgbm_withmacro_201901_sample.csv"
)

DATA = (
    ROOT / "data/processed/model_ready"
    / "28_monthly_transitions/late_development"
)

OUT = ROOT / "outputs/modeling/task33_loss"
OUT.mkdir(parents=True, exist_ok=True)

print("=== TASK 33.1: 12M CREDIT EXPOSURE ===", flush=True)

# 1. Load previously simulated 12M PD
pred = pd.read_csv(
    PD_FILE,
    dtype={
        "loan_id": "string",
        "_vintage_file": "string",
    },
)

assert len(pred) == 30000, "Expected 30,000 loan sample"

keys = ["_vintage_file", "loan_id"]

if pred.duplicated(keys).any():
    raise ValueError("Duplicate loan predictions")

if not (pred["asof_yyyymm"] == 201901).all():
    raise ValueError("Unexpected snapshot date")

# 2. Recover actual outstanding principal balance
balances = []

for fname, group in pred.groupby("_vintage_file"):

    if Path(fname).name != fname:
        raise ValueError("Unsafe vintage filename")

    path = DATA / fname

    table = pq.read_table(
        path,
        columns=["loan_id", "period_yyyymm", "upb_t"],
        filters=[("period_yyyymm", "=", 201901)],
    )

    frame = table.to_pandas()
    frame["loan_id"] = frame["loan_id"].astype("string")

    selected = frame.loc[
        frame["loan_id"].isin(group["loan_id"]),
        ["loan_id", "upb_t"],
    ].copy()

    selected["_vintage_file"] = fname

    if len(selected) != len(group):
        raise RuntimeError(
            f"{fname}: balance coverage mismatch "
            f"{len(selected)} vs {len(group)}"
        )

    if selected.duplicated(keys).any():
        raise RuntimeError(f"{fname}: duplicate balances")

    balances.append(selected)

    print(
        f"Matched {fname}: {len(selected):,}",
        flush=True,
    )

balance_df = pd.concat(balances, ignore_index=True)

result = pred.merge(
    balance_df,
    on=keys,
    how="left",
    validate="one_to_one",
)

# 3. Data quality
result["upb_t"] = pd.to_numeric(
    result["upb_t"], errors="coerce"
)

result["pd_12m"] = pd.to_numeric(
    result["pd_12m"], errors="coerce"
)

if result["upb_t"].isna().any():
    raise ValueError("Missing outstanding balances")

if (
    ~np.isfinite(result["upb_t"].to_numpy())
).any() or (result["upb_t"] < 0).any():
    raise ValueError("Invalid outstanding balances")

if (
    ~np.isfinite(result["pd_12m"].to_numpy())
).any() or (~result["pd_12m"].between(0, 1)).any():
    raise ValueError("Invalid 12M PD values")

# 4. Expected balance associated with first90 events
# This is not Expected Credit Loss.
# UPB as of 2019-01 is a proxy for future event-time EAD.

result["expected_first90_upb"] = (
    result["pd_12m"] * result["upb_t"]
)

result["observed_first90_upb_proxy"] = (
    result["first90_12m_actual"] * result["upb_t"]
)

# 5. Portfolio statistics
total_upb = result["upb_t"].sum()
expected_upb = result["expected_first90_upb"].sum()
observed_upb = result["observed_first90_upb_proxy"].sum()

expected_events = result["pd_12m"].sum()
actual_events = result["first90_12m_actual"].sum()

print("\n=== 12M CREDIT EXPOSURE RESULTS ===")

print("Loans:", f"{len(result):,}")
print("Total current UPB: $", f"{total_upb:,.2f}")

print(
    "Expected First90 loans:",
    f"{expected_events:,.2f}"
)
print(
    "Observed First90 loans:",
    f"{actual_events:,.0f}"
)

print(
    "Expected First90 exposure (as-of UPB): $",
    f"{expected_upb:,.2f}"
)

print(
    "Observed First90 exposure (as-of UPB): $",
    f"{observed_upb:,.2f}"
)

print(
    "Expected exposure / total UPB:",
    f"{expected_upb / total_upb:.4%}"
)

print("\n=== BY INITIAL STATE ===")

for state, group in result.groupby("state_t"):

    e = group["expected_first90_upb"].sum()
    a = group["observed_first90_upb_proxy"].sum()

    print(
        f"S{state}: loans={len(group):,}, "
        f"expected_exposure=${e:,.2f}, "
        f"observed_exposure=${a:,.2f}"
    )

# 6. Persist results
output = OUT / "12m_first90_exposure_201901_30k.csv"
result.to_csv(output, index=False)

print("\nSaved:", output)
print("No OOT data accessed.")
print("No training or 12M resimulation.")
print("Exposure uses as-of UPB, not actual default-time EAD.")
print("=== TASK 33.1 COMPLETE ===")
