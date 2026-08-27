"""Run exhaustive search on a sparse sequential benchmark dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from exhaustive_search import (
    DecisionLeaf,
    DecisionNode,
    DecisionTree,
    ExhaustiveSearchResult,
    exhaustive_search,
)
HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent / "min-error-datasets"
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from input_data import load_sparse_input, validate_reverse_phases  # noqa: E402

INPUT_ROOT = DATASET_ROOT / "inputs"
RESULTS_ROOT = HERE / "results" / "optimal_evaluation"
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def tree_to_dict(tree: DecisionTree) -> dict[str, object]:
    """Convert a decision tree to data that can be saved as JSON."""
    if isinstance(tree, DecisionLeaf):
        return {"declared_field": tree.declared_field}
    return {
        "pair": list(tree.pair),
        "location": tree.location,
        "pair_error": tree.pair_error,
        "discard_i": tree_to_dict(tree.discard_i),
        "discard_j": tree_to_dict(tree.discard_j),
    }


def validate_result(
    result: ExhaustiveSearchResult,
    K: int,
    M: int,
    budget: int,
) -> None:
    """Check the allocation, pair locations, tree, and final error."""
    if len(result.allocation) != M or sum(result.allocation) != budget:
        raise RuntimeError("exhaustive search returned an invalid allocation")
    expected_pairs = {(i, j) for i in range(K) for j in range(i + 1, K)}
    if set(result.pair_locations) != expected_pairs:
        raise RuntimeError("exhaustive search returned invalid pair locations")
    if any(not 0 <= location < M for location in result.pair_locations.values()):
        raise RuntimeError("exhaustive search returned an out-of-range location")
    if not 0.0 <= result.minimum_error <= 1.0:
        raise RuntimeError("exhaustive search returned an invalid error probability")
    _validate_tree(result.decision_tree, set(range(K)), M)


def _validate_tree(tree: DecisionTree, remaining: set[int], M: int) -> None:
    """Check that each tree branch removes one tested field."""
    if isinstance(tree, DecisionLeaf):
        if remaining != {tree.declared_field}:
            raise RuntimeError("decision-tree leaf declares the wrong field")
        return

    i, j = tree.pair
    if i not in remaining or j not in remaining or i >= j:
        raise RuntimeError("decision-tree node tests an invalid pair")
    if not 0 <= tree.location < M or not 0.0 <= tree.pair_error <= 0.5:
        raise RuntimeError("decision-tree node has invalid test data")
    _validate_tree(tree.discard_i, remaining - {i}, M)
    _validate_tree(tree.discard_j, remaining - {j}, M)


def evaluate_dataset(dataset: str) -> Path:
    """Evaluate every case and budget in one generated dataset."""
    if not DATASET_NAME.fullmatch(dataset):
        raise ValueError("dataset contains unsupported characters")
    dataset_directory = INPUT_ROOT / dataset
    manifest_path = dataset_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid manifest JSON: {manifest_path}") from error

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest must contain a non-empty cases list")

    records: list[dict[str, object]] = []

    # Evaluate every manifest case using each of its selected budgets.
    for case_number, case in enumerate(raw_cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"manifest case {case_number} must be an object")
        filename = case.get("file")
        budgets = case.get("sensor_budgets")
        minimum_sensors = case.get("minimum_sensors_for_zero_error")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"manifest case {case_number} has an invalid file")
        if not isinstance(budgets, list) or not budgets:
            raise ValueError(f"manifest case {case_number} has invalid budgets")
        if not isinstance(minimum_sensors, int):
            raise ValueError(f"manifest case {case_number} has invalid N_min")
        if any(not isinstance(budget, int) or budget >= minimum_sensors for budget in budgets):
            raise ValueError(f"every case {case_number} budget must be below N_min")

        K, M, phase_gaps, pair_errors = load_sparse_input(
            dataset_directory / filename
        )
        validate_reverse_phases(phase_gaps)
        if K != case.get("K") or M != case.get("M"):
            raise ValueError(f"case dimensions do not match manifest: {filename}")

        # Time and validate one exhaustive-search run for each budget.
        for budget in budgets:
            started = perf_counter()
            result = exhaustive_search(K, M, budget, pair_errors)
            elapsed = perf_counter() - started
            validate_result(result, K, M, budget)
            records.append(
                {
                    "case": Path(filename).stem,
                    "field_model": case.get("field_model"),
                    "spatial_dimension": case.get("spatial_dimension"),
                    "replicate": case.get("replicate"),
                    "K": K,
                    "M": M,
                    "minimum_sensors_for_zero_error": minimum_sensors,
                    "sensor_budget": budget,
                    "budget_fraction_of_N_min": budget / minimum_sensors,
                    "minimum_error": result.minimum_error,
                    "allocation": list(result.allocation),
                    "pair_locations": {
                        f"{i},{j}": location
                        for (i, j), location in result.pair_locations.items()
                    },
                    "decision_tree": tree_to_dict(result.decision_tree),
                    "seconds": elapsed,
                }
            )
            print(
                f"{filename} N={budget}: error={result.minimum_error:.8f}, "
                f"time={elapsed:.4f}s"
            )

    output_directory = RESULTS_ROOT / f"{dataset}_results_full" / "tables"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Keep full strategies in JSON and main numerical values in CSV.
    (output_directory / "results.json").write_text(
        json.dumps(
            {"dataset": dataset, "result_count": len(records), "results": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(output_directory / "summary.csv", records)
    return output_directory


def _write_summary_csv(path: Path, records: list[dict[str, object]]) -> None:
    """Write the main numerical results in a compact CSV file."""
    columns = [
        "case",
        "field_model",
        "spatial_dimension",
        "replicate",
        "K",
        "M",
        "minimum_sensors_for_zero_error",
        "sensor_budget",
        "budget_fraction_of_N_min",
        "minimum_error",
        "seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def _parse_arguments() -> argparse.Namespace:
    """Parse the dataset name from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=_dataset_name)
    return parser.parse_args()


def _dataset_name(value: str) -> str:
    """Validate a dataset directory name."""
    if not DATASET_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("dataset contains unsupported characters")
    return value


def main() -> int:
    """Run exhaustive search on the selected dataset."""
    arguments = _parse_arguments()
    output = evaluate_dataset(arguments.dataset)
    print(f"Saved results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
