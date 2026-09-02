"""Evaluate exact and greedy zero-error sensor allocation on one dataset.

Run from the repository root:

    python3 min-sensor-zero-error/scripts/evaluate_benchmarks.py --require-existing

There are no algorithm modes: every case is evaluated with both the exact
dynamic program and the greedy gain-based approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable
from math import log
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import List, Optional, Tuple

from greedy import greedy_perfect_pairwise_sensor_placement
from optimal import optimal_perfect_pairwise_sensor_placement
from phase_data import Thresholds, compute_nijm, load_phase_csv


HERE = Path(__file__).resolve().parents[1]
INPUT_ROOT = HERE / "inputs"
RESULTS_ROOT = HERE / "results" / "algorithm_comparison"
MAX_EXACT_FIELDS = 10
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SolverResult = Tuple[int, List[int], List[List[Optional[int]]]]
Solver = Callable[[int, int, Thresholds], SolverResult]


# Parse a command-line value that must be a positive integer.
def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


# Validate a dataset directory name supplied on the command line.
def _dataset_name(value: str) -> str:
    if not DATASET_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "may contain only letters, digits, underscores, and hyphens"
        )
    return value


# Define the command-line parameters for dataset evaluation.
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact and greedy algorithms on every dataset case."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="benchmark_1d",
        type=_dataset_name,
        help="Dataset directory below inputs/ (default: benchmark_1d).",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_integer,
        default=1,
        help="Deterministic runs per algorithm for mean timing (default: 1).",
    )
    return parser.parse_args()


# Load the dataset manifest and check its basic structure.
def _load_manifest(dataset_directory: Path) -> dict[str, object]:
    manifest_path = dataset_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read dataset manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in dataset manifest: {manifest_path}") from error

    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError(f"Manifest must contain a 'cases' list: {manifest_path}")
    return manifest


# Match each manifest entry to an existing CSV case file.
def _case_entries(
    dataset_directory: Path, manifest: dict[str, object]
) -> list[tuple[Path, dict[str, object]]]:
    entries: list[tuple[Path, dict[str, object]]] = []
    seen: set[str] = set()
    raw_cases = manifest["cases"]
    assert isinstance(raw_cases, list)
    for position, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict) or not isinstance(raw_case.get("file"), str):
            raise ValueError(f"Manifest case {position} must contain a string 'file'")
        filename = str(raw_case["file"])
        if Path(filename).name != filename or not filename.endswith(".csv"):
            raise ValueError(f"Invalid case filename in manifest: {filename}")
        if filename in seen:
            raise ValueError(f"Duplicate case filename in manifest: {filename}")
        seen.add(filename)
        path = dataset_directory / filename
        if not path.is_file():
            raise ValueError(f"Dataset case does not exist: {path}")
        entries.append((path, raw_case))

    unlisted = {path.name for path in dataset_directory.glob("*.csv")} - seen
    if unlisted:
        raise ValueError(
            "CSV files missing from manifest: " + ", ".join(sorted(unlisted))
        )
    if not entries:
        raise ValueError(f"Dataset contains no cases: {dataset_directory}")
    return entries


# Run a deterministic solver repeatedly and return its mean runtime.
def _run_repeatedly(
    solver: Solver,
    K: int,
    M: int,
    thresholds: Thresholds,
    repeats: int,
) -> tuple[SolverResult, float]:
    first_result: SolverResult | None = None
    started = perf_counter()
    for _ in range(repeats):
        result = solver(K, M, thresholds)
        if first_result is None:
            first_result = result
        elif result != first_result:
            raise RuntimeError("A deterministic solver returned inconsistent results")
    elapsed = (perf_counter() - started) / repeats
    assert first_result is not None
    return first_result, elapsed


# Check that a solver result covers every pair with a valid allocation.
def _validate_solution(
    label: str,
    result: SolverResult,
    K: int,
    M: int,
    thresholds: Thresholds,
) -> None:
    total, allocation, pair_locations = result
    if len(allocation) != M or total != sum(allocation):
        raise RuntimeError(f"{label} returned an inconsistent allocation")
    if len(pair_locations) != K or any(len(row) != K for row in pair_locations):
        raise RuntimeError(f"{label} returned an invalid pair-location matrix")
    for i in range(K):
        if pair_locations[i][i] is not None:
            raise RuntimeError(f"{label} pair-location diagonal must be None")
        for j in range(i + 1, K):
            location = pair_locations[i][j]
            if location != pair_locations[j][i] or not isinstance(location, int):
                raise RuntimeError(f"{label} returned an invalid location for pair {(i, j)}")
            if not 0 <= location < M:
                raise RuntimeError(f"{label} returned an out-of-range location")
            threshold = thresholds[(i, j)][location]
            if threshold is None or allocation[location] < threshold:
                raise RuntimeError(f"{label} did not cover pair {(i, j)}")


# Convert one solver result into a JSON-ready record.
def _solver_record(result: SolverResult, seconds: float) -> dict[str, object]:
    total, allocation, pair_locations = result
    return {
        "minimum_sensors": total,
        "allocation": allocation,
        "pair_locations": pair_locations,
        "mean_seconds": seconds,
    }


# Run and validate both algorithms for one benchmark case.
def _evaluate_case(
    case_path: Path,
    metadata: dict[str, object],
    repeats: int,
) -> dict[str, object]:
    K, M, phase_gaps = load_phase_csv(case_path)
    if K > MAX_EXACT_FIELDS:
        raise ValueError(
            f"{case_path.name} has K={K}; evaluation always runs the exact algorithm "
            f"and therefore supports K <= {MAX_EXACT_FIELDS}"
        )
    thresholds = compute_nijm(K, M, phase_gaps)

    optimal_result, optimal_seconds = _run_repeatedly(
        optimal_perfect_pairwise_sensor_placement, K, M, thresholds, repeats
    )
    greedy_result, greedy_seconds = _run_repeatedly(
        greedy_perfect_pairwise_sensor_placement, K, M, thresholds, repeats
    )
    _validate_solution("optimal", optimal_result, K, M, thresholds)
    _validate_solution("greedy", greedy_result, K, M, thresholds)

    optimal_total = optimal_result[0]
    greedy_total = greedy_result[0]
    pair_count = K * (K - 1) // 2
    ratio = greedy_total / optimal_total
    approximation_factor = 1 + log(pair_count)

    # Confirm that the observed ratio respects the theoretical guarantee.
    if ratio > approximation_factor + 1e-12:
        raise RuntimeError(
            f"Greedy result violates the paper's approximation bound in {case_path.name}"
        )

    return {
        "case": case_path.stem,
        "field_model": metadata.get("field_model", "unknown"),
        "spatial_dimension": metadata.get("spatial_dimension"),
        "replicate": metadata.get("replicate"),
        "generation_seed": metadata.get("generation_seed"),
        "phase_scale_c": metadata.get("phase_scale_c"),
        "K": K,
        "M": M,
        "pair_count": pair_count,
        "optimal": _solver_record(optimal_result, optimal_seconds),
        "greedy": _solver_record(greedy_result, greedy_seconds),
        "comparison": {
            "additional_sensors": greedy_total - optimal_total,
            "greedy_to_optimal_ratio": ratio,
            "theorem_factor": approximation_factor,
            "bound_utilization": ratio / approximation_factor,
        },
    }


# Flatten a detailed case record into one summary-table row.
def _summary_row(record: dict[str, object]) -> dict[str, object]:
    optimal = record["optimal"]
    greedy = record["greedy"]
    comparison = record["comparison"]
    assert isinstance(optimal, dict)
    assert isinstance(greedy, dict)
    assert isinstance(comparison, dict)
    return {
        "dataset": record.get("dataset"),
        "case": record["case"],
        "field_model": record["field_model"],
        "spatial_dimension": record.get("spatial_dimension"),
        "replicate": record.get("replicate"),
        "generation_seed": record.get("generation_seed"),
        "phase_scale_c": record.get("phase_scale_c"),
        "K": record["K"],
        "M": record["M"],
        "N_optimal": optimal["minimum_sensors"],
        "N_greedy": greedy["minimum_sensors"],
        "additional_sensors": comparison["additional_sensors"],
        "greedy_to_optimal_ratio": comparison["greedy_to_optimal_ratio"],
        "theorem_factor": comparison["theorem_factor"],
        "bound_utilization": comparison["bound_utilization"],
        "optimal_seconds": optimal["mean_seconds"],
        "greedy_seconds": greedy["mean_seconds"],
    }


def summarize_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return the compact result rows used by CSV and notebook displays."""
    return [_summary_row(record) for record in records]


# Write detailed JSON results and the compact CSV summary.
def _write_outputs(
    dataset_name: str,
    manifest: dict[str, object],
    records: list[dict[str, object]],
    repeats: int,
) -> Path:
    output_directory = RESULTS_ROOT / f"{dataset_name}_results_full" / "tables"
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": dataset_name,
        "timing_repetitions": repeats,
        "phase_data": manifest.get("phase_data"),
        "cases": records,
    }
    (output_directory / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    rows = summarize_records(records)
    with (output_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_directory


# Print a readable case-by-case summary for command-line runs.
def _print_summary(dataset_name: str, records: list[dict[str, object]]) -> None:
    rows = summarize_records(records)
    print(f"\nDataset: {dataset_name} ({len(rows)} cases; exact + greedy)")
    print("case         K  M  optimal  greedy  ratio   optimal(s)  greedy(s)")
    for row in rows:
        print(
            f"{str(row['case']):<12} {int(row['K']):>2} {int(row['M']):>2} "
            f"{int(row['N_optimal']):>8} {int(row['N_greedy']):>7} "
            f"{float(row['greedy_to_optimal_ratio']):>6.3f} "
            f"{float(row['optimal_seconds']):>11.6f} "
            f"{float(row['greedy_seconds']):>10.6f}"
        )
    print(
        "Mean greedy/optimal ratio: "
        f"{fmean(float(row['greedy_to_optimal_ratio']) for row in rows):.3f}"
    )


def evaluate_dataset(
    dataset_name: str = "benchmark_1d", repeats: int = 1
) -> tuple[list[dict[str, object]], Path]:
    """Run both algorithms for every case and write JSON/CSV results."""
    if not isinstance(dataset_name, str) or not DATASET_NAME.fullmatch(dataset_name):
        raise ValueError(
            "dataset_name may contain only letters, digits, underscores, and hyphens"
        )
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    dataset_directory = INPUT_ROOT / dataset_name
    if not dataset_directory.is_dir():
        raise ValueError(f"Dataset does not exist: {dataset_directory}")

    manifest = _load_manifest(dataset_directory)
    if manifest.get("dataset") != dataset_name:
        raise ValueError(
            "Manifest dataset name does not match its directory: "
            f"expected {dataset_name!r}, found {manifest.get('dataset')!r}"
        )
    entries = _case_entries(dataset_directory, manifest)
    records = [
        _evaluate_case(case_path, metadata, repeats)
        for case_path, metadata in entries
    ]
    for record in records:
        record["dataset"] = dataset_name
    output_directory = _write_outputs(dataset_name, manifest, records, repeats)
    return records, output_directory


# Parse arguments, run the evaluation, and report the output directory.
def main() -> int:
    arguments = _parse_arguments()

    try:
        records, output_directory = evaluate_dataset(
            arguments.dataset, arguments.repeats
        )
    except (ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    _print_summary(arguments.dataset, records)
    print(f"Results: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
