
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from zipfile import ZipFile
import argparse
import csv
import re

import pyarrow as pa
import pyarrow.parquet as pq


BASE = Path.home() / "Desktop/credit-portfolio-data"

# Freddie Mac Release 47 Performance column positions (0-based)
COLS = [
    ("loan_id", 0),
    ("period_yyyymm", 1),
    ("delinquency_raw", 3),
    ("modification_raw", 7),
    ("zero_balance_raw", 8),
    ("deferred_upb_raw", 11),
    ("payment_deferral_raw", 24),
    ("eltv_raw", 25),
    ("disaster_raw", 28),
    ("assistance_raw", 29),
]

SCHEMA = pa.schema([
    pa.field(
        name,
        pa.int32() if name == "period_yyyymm" else pa.string()
    )
    for name, _ in COLS
])

TXT = re.compile(
    r"(?:sample_(?:perf|svcg)|perf)_(20\d{2})\.txt$",
    re.I,
)
ZIP = re.compile(r"sample_(20\d{2})\.zip$", re.I)


def locate(folder):
    found = defaultdict(list)

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue

        if (match := TXT.search(path.name)):
            found[int(match[1])].append(("text", path))
        elif (match := ZIP.search(path.name)):
            found[int(match[1])].append(("zip", path))

    return {
        year: sorted(
            candidates,
            key=lambda x: (x[0] != "text", len(str(x[1])))
        )[0]
        for year, candidates in found.items()
    }


@contextmanager
def lines_from(kind, path, year):
    if kind == "text":
        with path.open("rb") as f:
            yield f
    else:
        with ZipFile(path) as archive:
            matches = [
                name for name in archive.namelist()
                if TXT.search(Path(name).name)
                and int(TXT.search(Path(name).name)[1]) == year
                and "__MACOSX" not in name
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"{path.name}: expected one Performance TXT, "
                    f"found {matches}"
                )

            with archive.open(matches[0]) as f:
                yield f


def write_batch(buffer, writer, path):
    if writer is None:
        writer = pq.ParquetWriter(
            path,
            SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )

    writer.write_table(
        pa.table(buffer, schema=SCHEMA)
    )

    for column in buffer.values():
        column.clear()

    return writer


def process(year, source, output, monthly):
    kind, path = source

    target = output / f"assistance_vintage_{year}.parquet"
    temp = output / f".assistance_vintage_{year}.partial.parquet"

    temp.unlink(missing_ok=True)

    buffer = {name: [] for name, _ in COLS}

    writer = None
    kept = 0
    scanned = 0

    try:
        with lines_from(kind, path, year) as stream:
            for line in stream:
                scanned += 1
                values = line.rstrip(b"\r\n").split(b"|")

                if len(values) != 35:
                    raise ValueError(
                        f"{path.name}: row {scanned} has "
                        f"{len(values)} columns; expected 35"
                    )

                month_raw = values[1].strip()

                # Exclude observations outside the development period.
                if month_raw < b"201401" or month_raw > b"202012":
                    continue

                if (
                    len(month_raw) != 6
                    or not month_raw.isdigit()
                    or not 1 <= int(month_raw[-2:]) <= 12
                ):
                    raise ValueError(
                        f"Invalid reporting month at row {scanned}"
                    )

                if not values[0].strip():
                    raise ValueError(
                        f"Blank loan ID at row {scanned}"
                    )

                month = int(month_raw)

                for name, index in COLS:
                    value = (
                        month if name == "period_yyyymm"
                        else values[index]
                        .decode("utf-8", "replace")
                        .strip()
                    )
                    buffer[name].append(value)

                assistance = (
                    values[29].decode("ascii", "replace")
                    .strip().upper() or "BLANK"
                )

                deferral = values[24].strip().upper()
                disaster = values[28].strip().upper()
                delinquency = values[3].strip().upper()

                counts = monthly[(month, assistance)]

                counts[0] += 1
                counts[1] += int(
                    delinquency.isdigit()
                    and int(delinquency) >= 3
                )
                counts[2] += int(
                    deferral in (b"Y", b"C", b"P")
                )
                counts[3] += int(disaster == b"Y")

                kept += 1

                if len(buffer["loan_id"]) == 100_000:
                    writer = write_batch(
                        buffer, writer, temp
                    )

            if buffer["loan_id"]:
                writer = write_batch(
                    buffer, writer, temp
                )

        if writer is not None:
            writer.close()
            writer = None
            temp.replace(target)

    except BaseException:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)
        raise

    print(
        f"Vintage {year}: "
        f"scanned={scanned:,}, "
        f"extracted={kept:,}",
        flush=True,
    )

    return kept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "outputs/diagnostics/task31_5_assistance",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
    )

    args = parser.parse_args()

    source_root = args.source_dir.expanduser()

    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    sources = locate(source_root)
    vintages = range(2006, 2021)

    missing = [
        year for year in vintages
        if year not in sources
    ]

    print("\n=== RAW FILE INVENTORY ===")
    print(
        "Found:",
        [year for year in vintages if year in sources],
    )
    print("Missing:", missing)

    if args.inventory_only or missing:
        if missing:
            print(
                "STOP: Some source files are missing. "
                "Check --source-dir."
            )
        return

    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    monthly = defaultdict(lambda: [0, 0, 0, 0])
    total = 0

    print("\n=== EXTRACTING 2014-2020 ===", flush=True)

    for year in vintages:
        total += process(
            year, sources[year], output, monthly
        )

    summary = output / "monthly_assistance_summary.csv"

    with summary.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "period_yyyymm",
            "assistance_code",
            "loan_months",
            "90plus_loan_months",
            "payment_deferral_loan_months",
            "disaster_flag_loan_months",
        ])

        for (month, code), counts in sorted(monthly.items()):
            writer.writerow([month, code, *counts])

    statuses = defaultdict(int)

    for (_, code), counts in monthly.items():
        statuses[code] += counts[0]

    print("\n=== SUMMARY ===")
    print("Extracted loan-month rows:", f"{total:,}")

    for code, count in sorted(
        statuses.items(),
        key=lambda item: -item[1],
    ):
        print(f"{code}: {count:,}")

    print("\nMonthly summary:", summary)
    print("Parquet directory:", output)
    print("Original files unchanged.")
    print("No 2021-2023 observations saved.")


if __name__ == "__main__":
    main()
