"""Run the complete, pre-specified paper evaluation workflow."""

from __future__ import annotations

import argparse

from eval import INPUT_ROOT, OUTPUT_ROOT, evaluate_dataset
from generate_paper_studies import STUDIES, generate_studies
from paper_analysis import analyze_paper_studies


DATASETS = [str(design["dataset"]) for design in STUDIES.values()]


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def run_paper_evaluation(
    *,
    repeats: int = 1,
    bootstrap_iterations: int = 2_000,
    generate_missing: bool = True,
) -> dict[str, object]:
    """Generate missing studies, evaluate both solvers, and analyze results."""
    missing_names = [
        name
        for name, design in STUDIES.items()
        if not (INPUT_ROOT / str(design["dataset"]) / "manifest.json").is_file()
    ]
    if missing_names and not generate_missing:
        missing = ", ".join(missing_names)
        raise ValueError(f"Missing paper studies: {missing}")
    if missing_names:
        generate_studies(missing_names)

    records_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset in DATASETS:
        print(f"Evaluating {dataset} ...", flush=True)
        records, output_directory = evaluate_dataset(dataset, repeats)
        records_by_dataset[dataset] = records
        print(f"  {len(records)} cases -> {output_directory}", flush=True)

    analysis_directory = OUTPUT_ROOT / "paper-analysis"
    analysis = analyze_paper_studies(
        records_by_dataset,
        analysis_directory,
        bootstrap_iterations=bootstrap_iterations,
    )
    print(f"Paper tables and figures -> {analysis_directory}")
    return analysis


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, evaluate, and analyze all pre-specified paper studies."
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
        help="Fail instead of generating a paper dataset that is missing.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        run_paper_evaluation(
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
