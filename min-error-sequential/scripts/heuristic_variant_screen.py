"""Prototype stronger sequential heuristics on saved Problem 2 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter

HERE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HERE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from actual_error_analysis import actual_tree_error
from exhaustive_search import exhaustive_search
from greedy import greedy
from heuristic import (
    DecisionTree,
    _build_tree_with_beam_search,
    _compute_pair_errors,
    _initial_allocation,
    _tree_error,
    heuristic,
)
from input_data import PairErrorData, load_sparse_input

INPUT_ROOT = HERE / "inputs"


@dataclass(frozen=True)
class CandidateResult:
    """The allocation, tree, and surrogate score from one candidate heuristic."""

    allocation: tuple[int, ...]
    decision_tree: DecisionTree
    surrogate_error: float
    iterations: int


def best_improvement_local_search(
    K: int,
    M: int,
    N: int,
    pair_errors: PairErrorData,
    *,
    beam_width: int = 8,
    max_iterations: int = 50,
    improvement_threshold: float = 1e-5,
    include_greedy_start: bool = True,
) -> CandidateResult:
    """Coordinate descent over one-sensor moves using the full tree objective."""
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    starts = [_initial_allocation(M, N, "near_uniform", 0)]
    if include_greedy_start:
        starts.append(greedy(K, M, N, pair_errors).allocation)

    best = min(
        (
            _evaluate_allocation(K, M, allocation, pair_errors, beam_width)
            for allocation in starts
        ),
        key=lambda result: result.surrogate_error,
    )

    for iteration in range(1, max_iterations + 1):
        best_move = best
        for source in range(M):
            if best.allocation[source] == 0:
                continue
            for destination in range(M):
                if destination == source:
                    continue
                candidate = list(best.allocation)
                candidate[source] -= 1
                candidate[destination] += 1
                result = _evaluate_allocation(
                    K,
                    M,
                    tuple(candidate),
                    pair_errors,
                    beam_width,
                )
                if result.surrogate_error < best_move.surrogate_error:
                    best_move = result

        if (
            best_move.surrogate_error
            >= best.surrogate_error - improvement_threshold
        ):
            return CandidateResult(
                allocation=best.allocation,
                decision_tree=best.decision_tree,
                surrogate_error=best.surrogate_error,
                iterations=iteration - 1,
            )
        best = best_move

    return CandidateResult(
        allocation=best.allocation,
        decision_tree=best.decision_tree,
        surrogate_error=best.surrogate_error,
        iterations=max_iterations,
    )


def _evaluate_allocation(
    K: int,
    M: int,
    allocation: tuple[int, ...],
    pair_errors: PairErrorData,
    beam_width: int,
) -> CandidateResult:
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    pair_locations, current_pair_errors = _compute_pair_errors(
        pairs,
        M,
        allocation,
        pair_errors,
    )
    tree = _build_tree_with_beam_search(
        K,
        pair_locations,
        current_pair_errors,
        beam_width,
    )
    return CandidateResult(
        allocation=allocation,
        decision_tree=tree,
        surrogate_error=_tree_error(tree, K),
        iterations=0,
    )


def run_trial(
    dataset: str,
    *,
    replicates_per_cell: int,
    max_cases: int | None,
    beam_width: int,
    heuristic_initialization: str,
    heuristic_max_iterations: int,
    heuristic_shortlist_size: int,
    heuristic_include_greedy_start: bool,
    run_full_candidate: bool,
    candidate_include_greedy_start: bool,
) -> None:
    """Run the experiment on a small existing-data subset."""
    manifest = json.loads((INPUT_ROOT / dataset / "manifest.json").read_text())
    cases = [
        case
        for case in manifest["cases"]
        if int(case["replicate"]) <= replicates_per_cell
    ]
    if max_cases is not None:
        cases = cases[:max_cases]

    records: list[dict[str, float | int | str]] = []
    for case_number, case in enumerate(cases, start=1):
        K, M, phases, errors = load_sparse_input(
            INPUT_ROOT / dataset / str(case["file"])
        )
        for budget_level, budget in enumerate(case["sensor_budgets"], start=1):
            budget = int(budget)
            started = perf_counter()
            optimal = exhaustive_search(K, M, budget, errors)
            optimal_seconds = perf_counter() - started

            started = perf_counter()
            greedy_result = greedy(K, M, budget, errors)
            greedy_seconds = perf_counter() - started

            started = perf_counter()
            current = heuristic(
                K,
                M,
                budget,
                errors,
                beam_width=beam_width,
                initialization=heuristic_initialization,
                max_iterations=heuristic_max_iterations,
                include_greedy_start=heuristic_include_greedy_start,
                move_shortlist_size=heuristic_shortlist_size,
            )
            current_seconds = perf_counter() - started

            if run_full_candidate:
                started = perf_counter()
                candidate = best_improvement_local_search(
                    K,
                    M,
                    budget,
                    errors,
                    beam_width=beam_width,
                    include_greedy_start=candidate_include_greedy_start,
                )
                candidate_seconds = perf_counter() - started
                candidate_surrogate = candidate.surrogate_error
                candidate_actual = actual_tree_error(
                    candidate.decision_tree,
                    K,
                    phases,
                    candidate.allocation,
                )
                candidate_iterations = candidate.iterations
            else:
                candidate_seconds = 0.0
                candidate_surrogate = current.error_probability
                candidate_actual = actual_tree_error(
                    current.decision_tree,
                    K,
                    phases,
                    current.allocation,
                )
                candidate_iterations = current.iterations_run

            records.append(
                {
                    "dataset": dataset,
                    "K": K,
                    "M": M,
                    "budget_level": budget_level,
                    "rho": budget / int(case["minimum_sensors_for_zero_error"]),
                    "optimal_surrogate": optimal.minimum_error,
                    "greedy_surrogate": greedy_result.error_probability,
                    "current_surrogate": current.error_probability,
                    "candidate_surrogate": candidate_surrogate,
                    "optimal_actual": actual_tree_error(
                        optimal.decision_tree, K, phases, optimal.allocation
                    ),
                    "greedy_actual": actual_tree_error(
                        greedy_result.decision_tree, K, phases, greedy_result.allocation
                    ),
                    "current_actual": actual_tree_error(
                        current.decision_tree, K, phases, current.allocation
                    ),
                    "candidate_actual": candidate_actual,
                    "greedy_seconds": greedy_seconds,
                    "current_seconds": current_seconds,
                    "candidate_seconds": candidate_seconds,
                    "candidate_iterations": candidate_iterations,
                    "optimal_seconds": optimal_seconds,
                }
            )
        print(f"{dataset}: completed case {case_number}/{len(cases)}", flush=True)

    _print_summary(records)


def _print_summary(records: list[dict[str, float | int | str]]) -> None:
    print("\nOverall means")
    for key in ("surrogate", "actual"):
        print(
            key,
            "optimal",
            fmean(float(record[f"optimal_{key}"]) for record in records),
            "greedy",
            fmean(float(record[f"greedy_{key}"]) for record in records),
            "current",
            fmean(float(record[f"current_{key}"]) for record in records),
            "candidate",
            fmean(float(record[f"candidate_{key}"]) for record in records),
        )

    print("\nMean runtime seconds")
    print(
        "greedy",
        fmean(float(record["greedy_seconds"]) for record in records),
        "current",
        fmean(float(record["current_seconds"]) for record in records),
        "candidate_full",
        (
            fmean(float(record["candidate_seconds"]) for record in records)
            if any(float(record["candidate_seconds"]) > 0.0 for record in records)
            else "skipped"
        ),
        "optimal",
        fmean(float(record["optimal_seconds"]) for record in records),
    )
    print(
        "current/greedy runtime ratio",
        fmean(
            float(record["current_seconds"]) / float(record["greedy_seconds"])
            for record in records
            if float(record["greedy_seconds"]) > 0.0
        ),
    )

    print("\nBy budget level")
    for level in sorted({int(record["budget_level"]) for record in records}):
        selected = [
            record
            for record in records
            if int(record["budget_level"]) == level
        ]
        print(
            "level",
            level,
            "rho",
            fmean(float(record["rho"]) for record in selected),
        )
        for key in ("surrogate", "actual"):
            print(
                " ",
                key,
                "opt",
                fmean(float(record[f"optimal_{key}"]) for record in selected),
                "greedy",
                fmean(float(record[f"greedy_{key}"]) for record in selected),
                "curr",
                fmean(float(record[f"current_{key}"]) for record in selected),
                "cand",
                fmean(float(record[f"candidate_{key}"]) for record in selected),
            )

    wins = {
        key: sum(
            float(record[f"optimal_{key}"])
            <= float(record[f"candidate_{key}"])
            <= float(record[f"greedy_{key}"])
            for record in records
        )
        for key in ("surrogate", "actual")
    }
    print("\nOrdering rate Optimal <= Candidate <= Greedy")
    for key, count in wins.items():
        print(key, count, "/", len(records), count / len(records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="compact_benchmark_1d")
    parser.add_argument("--replicates-per-cell", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--heuristic-initialization", default="greedy")
    parser.add_argument("--heuristic-max-iterations", type=int, default=3)
    parser.add_argument("--heuristic-shortlist-size", type=int, default=1)
    parser.add_argument(
        "--no-heuristic-greedy-start",
        action="store_false",
        dest="heuristic_include_greedy_start",
        help="Do not use the greedy allocation as a heuristic start.",
    )
    parser.add_argument(
        "--skip-full-candidate",
        action="store_false",
        dest="run_full_candidate",
        help="Do not run the expensive all-moves local-search prototype.",
    )
    parser.add_argument(
        "--no-candidate-greedy-start",
        action="store_false",
        dest="candidate_include_greedy_start",
        help="Use only the requested initialization for the prototype candidate.",
    )
    args = parser.parse_args()
    run_trial(
        args.dataset,
        replicates_per_cell=args.replicates_per_cell,
        max_cases=args.max_cases,
        beam_width=args.beam_width,
        heuristic_initialization=args.heuristic_initialization,
        heuristic_max_iterations=args.heuristic_max_iterations,
        heuristic_shortlist_size=args.heuristic_shortlist_size,
        heuristic_include_greedy_start=args.heuristic_include_greedy_start,
        run_full_candidate=args.run_full_candidate,
        candidate_include_greedy_start=args.candidate_include_greedy_start,
    )


if __name__ == "__main__":
    main()
