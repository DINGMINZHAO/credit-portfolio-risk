
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

target = DATA / "03_loan_events_2006_2022.parquet"

if not target.exists():
    raise FileNotFoundError(target)

pf = pq.ParquetFile(target)
columns = pf.schema_arrow.names

print("\n=== 1. MAIN DATASET ===")
print(f"File: {target.name}")
print(f"Rows: {pf.metadata.num_rows:,}")

print("\n=== 2. AVAILABLE COLUMNS ===")
for name in columns:
    print(name)

if "vintage" in columns:
    vintages = pq.read_table(
        target, columns=["vintage"]
    ).to_pandas()

    print("\n=== 3. VINTAGE COVERAGE ===")
    print(
        vintages["vintage"]
        .value_counts()
        .sort_index()
        .to_string()
    )

print("\n=== 4. OTHER RELEVANT PARQUET FILES ===")
keywords = (
    "loss", "lgd", "event", "followup",
    "resolution", "performance"
)

for path in sorted(DATA.rglob("*.parquet")):
    if path == target:
        continue
    if any(k in path.name.lower() for k in keywords):
        meta = pq.ParquetFile(path).metadata
        print(
            f"{path.relative_to(DATA)} "
            f"| rows={meta.num_rows:,}"
        )

print("\nInventory complete.")
