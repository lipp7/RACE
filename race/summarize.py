"""Summarize complete evaluations as mean and sample standard deviation."""
import argparse
import statistics
from .io import project_root, relative_path, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="results/summary.json")
    args = parser.parse_args()
    project_root()
    values = []
    records = []
    for name in args.runs:
        folder = relative_path(name)
        result = read_json(folder / "evaluation_full.json")
        values.append(100 * result["accuracy"])
        records.append(dict(run=str(folder), accuracy_percent=values[-1], test_n=result["n"]))
    summary = dict(mean=statistics.mean(values),
                   sample_std=statistics.stdev(values) if len(values) > 1 else None,
                   runs=records)
    write_json(relative_path(args.output), summary)
    print(summary)


if __name__ == "__main__":
    main()
