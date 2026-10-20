
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

YEARS = list(range(2006,2023))
RAW_DIR = Path("data/raw")
OUTPUT = Path("outputs/01_inventory.csv")


def read_rows(archive, filename):
    with archive.open(filename) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as f:
            yield from csv.reader(f, delimiter="|")


def audit_year(year):
    path = RAW_DIR / f"sample_{year}.zip"
    orig_ids = set()
    states = Counter()

    loan_months = Counter()
    dpd90 = set()
    dpd180 = set()
    reo = set()
    exits = defaultdict(set)
    loss_observed = set()
    positive_loss = set()
    negative_loss = set()
    invalid_loss = 0

    first_period = "999999"
    last_period = "000000"

    with ZipFile(path) as archive:
        # Origination: 31 columns
        for row in read_rows(archive, f"sample_orig_{year}.txt"):
            if len(row) != 31:
                raise ValueError(f"{year}: Invalid origination layout")

            orig_ids.add(row[19])
            states[row[16]] += 1

        # Monthly performance: 35 columns
        for row in read_rows(archive, f"sample_perf_{year}.txt"):
            if len(row) != 35:
                raise ValueError(f"{year}: Invalid performance layout")

            loan_id = row[0]
            period = row[1]
            status = row[3].strip()
            exit_code = row[8].strip()
            loss_raw = row[21].strip()

            loan_months[loan_id] += 1
            first_period = min(first_period, period)
            last_period = max(last_period, period)

            if status.isdigit():
                if int(status) >= 3:
                    dpd90.add(loan_id)
                if int(status) >= 6:
                    dpd180.add(loan_id)

            if status == "RA":
                reo.add(loan_id)

            if exit_code:
                exits[exit_code].add(loan_id)

            if loss_raw:
                try:
                    loss = float(loss_raw)
                    loss_observed.add(loan_id)

                    if loss > 0:
                        positive_loss.add(loan_id)
                    elif loss < 0:
                        negative_loss.add(loan_id)
                except ValueError:
                    invalid_loss += 1

    n_loans = len(orig_ids)

    result = {
        "year": year,
        "loans": n_loans,
        "performance_rows": sum(loan_months.values()),
        "performance_loans": len(loan_months),
        "unmatched_loans": len(set(loan_months) - orig_ids),
        "avg_months_per_loan": round(
            sum(loan_months.values()) / max(len(loan_months), 1), 1
        ),
        "first_period": first_period,
        "last_period": last_period,
        "ever_90dpd": len(dpd90),
        "ever_180dpd": len(dpd180),
        "reo": len(reo),
        "credit_dispositions": len(
            set().union(*(exits[c] for c in ("02", "03", "09")))
        ),
        "prepaid": len(exits["01"]),
        "loss_observed": len(loss_observed),
        "positive_loss": len(positive_loss),
        "negative_loss": len(negative_loss),
        "invalid_loss_rows": invalid_loss,
        "top_state": states.most_common(1)[0][0] if states else "",
    }

    return result


def main():
    results = []

    for year in YEARS:
        print(f"Processing {year}...", flush=True)
        result = audit_year(year)
        results.append(result)

        print(
            f"  Loans: {result['loans']:,} | "
            f"Rows: {result['performance_rows']:,} | "
            f"90+ DPD: {result['ever_90dpd']:,} | "
            f"Positive Loss: {result['positive_loss']:,}",
            flush=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
