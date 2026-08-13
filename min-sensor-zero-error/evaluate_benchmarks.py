"""Generate, evaluate, and analyze both spatial benchmark datasets."""

from __future__ import annotations

import argparse

from eval import INPUT_ROOT, RESULTS_ROOT, evaluate_dataset
from generate_benchmarks import BENCHMARKS, generate_benchmarks
from paper_analysis import analyze_benchmarks


DATASETS = [str(design["dataset"]) for design in BENCHMARKS.values()]


# Parse a command-line value that must be a positive integer.
def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def run_benchmark_evaluation(
    *,
    repeats: int = 1,
    bootstrap_iterations: int = 2_000,
    generate_missing: bool = True,
) -> dict[str, object]:
    """Generate missing benchmarks, run both solvers, and analyze results."""
    missing_names = [
        name
        for name, design in BENCHMARKS.items()
        if not (INPUT_ROOT / str(design["dataset"]) / "manifest.json").is_file()
    ]
    if missing_names and not generate_missing:
        raise ValueError("Missing benchmarks: " + ", ".join(missing_names))
    if missing_names:
        generate_benchmarks(missing_names)

    records_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset in DATASETS:
        print(f"Evaluating {dataset} ...", flush=True)
        records, output_directory = evaluate_dataset(dataset, repeats)
        records_by_dataset[dataset] = records
        print(f"  {len(records)} cases -> {output_directory}", flush=True)

    dataset_names = "_".join(DATASETS)
    analysis_directory = RESULTS_ROOT / f"results_summary_{dataset_names}_full"
    analysis = analyze_benchmarks(
        records_by_dataset,
        analysis_directory,
        bootstrap_iterations=bootstrap_iterations,
    )
    print(f"Benchmark tables and figures -> {analysis_directory}")
    return analysis


# Define command-line options for the complete benchmark pipeline.
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, evaluate, and analyze both benchmark datasets."
    )
    parser.add_argument(
        "--repeats",
        type=_positive_integer,
        default=1,
        help="Solver repetitions used only for runtime averaging (default: 1).",
    )
    parser.add_argument(
        "--bootstrap",
        type=_positive_integer,
        default=2_000,
        help="Bootstrap resamples for mean-ratio intervals (default: 2000).",
    )
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Fail instead of generating a missing benchmark.",
    )
    return parser.parse_args()


# Run the benchmark pipeline and return a process status code.
def main() -> int:
    arguments = _parse_arguments()
    try:
        run_benchmark_evaluation(
            repeats=arguments.repeats,
            bootstrap_iterations=arguments.bootstrap,
            generate_missing=not arguments.require_existing,
        )
    except (ValueError, RuntimeError) as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
