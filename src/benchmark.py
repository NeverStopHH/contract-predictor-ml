import csv
from pathlib import Path

from src.train_dt_rf import run as run_dt_rf
from src.train_nn import run as run_nn


def write_benchmark_csv(rows: list[dict], filename: str = "benchmark.csv") -> None:
    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename

    # Determine all columns that appear in rows
    fieldnames = sorted({k for r in rows for k in r.keys()})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_path = "data/Trainingsdatensatz.xlsx"
    test_path = "data/Testdatensatz.xlsx"

    results = []

    # DT + RF
    results.extend(run_dt_rf(train_path, test_path))

    # NN
    results.append(run_nn(train_path, test_path, steps=5000))

    write_benchmark_csv(results)

    print("\nSaved benchmark table to reports/metrics/benchmark.csv")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()