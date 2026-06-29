"""List benchmark ids that do not have candidate-cache JSON files yet."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_benchmarks(path: Path | None, text: str | None) -> list[str]:
    values: list[str] = []
    if path is not None:
        values.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    if text:
        values.extend(parse_csv_values(text))
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks-file", type=Path, default=None)
    parser.add_argument("--benchmarks", default=None)
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--separator", default=",")
    args = parser.parse_args()

    benchmarks = read_benchmarks(args.benchmarks_file, args.benchmarks)
    missing = [benchmark for benchmark in benchmarks if not (args.candidate_cache_dir / f"{benchmark}.json").exists()]
    print(args.separator.join(missing))


if __name__ == "__main__":
    main()

