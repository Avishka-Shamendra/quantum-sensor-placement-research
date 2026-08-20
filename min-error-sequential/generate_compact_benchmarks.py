"""Generate the compact one- and two-dimensional benchmark datasets."""

from __future__ import annotations

import argparse

from generate_inputs import generate_factorial_dataset


MODELS = ["gaussian", "sinusoidal", "quadratic"]
K_VALUES = [3, 4, 5]
M_VALUES = [3, 4]
REPLICATES_PER_CELL = 140

# This cap limits exhaustive-search runtime. It is not a problem constraint.
EXHAUSTIVE_SEARCH_BUDGET_CAP = 12

# Separate seeds make the 1D and 2D datasets reproducible and independent.
BENCHMARKS: dict[str, dict[str, object]] = {
    "1d": {
        "dataset": "compact_benchmark_1d",
        "dimension": 1,
        "seed": 20260812,
    },
    "2d": {
        "dataset": "compact_benchmark_2d",
        "dimension": 2,
        "seed": 20260813,
    },
}


def generate_compact_benchmarks(
    names: list[str], overwrite: bool = False
) -> list[str]:
    """Generate selected compact benchmarks and return their directory names."""
    generated: list[str] = []
    for name in names:
        design = BENCHMARKS[name]
        dataset = str(design["dataset"])
        target = generate_factorial_dataset(
            dataset,
            dimension=int(design["dimension"]),
            models=MODELS,
            k_values=K_VALUES,
            m_values=M_VALUES,
            replicates=REPLICATES_PER_CELL,
            cases_per_model=None,
            phase_scale=1.0,
            budget_fractions=[0.25, 0.5, 0.75, 1.0],
            budget_cap=EXHAUSTIVE_SEARCH_BUDGET_CAP,
            samples_per_location=3,
            seed=int(design["seed"]),
            description=(
                f"Compact {name.upper()} benchmark for exact minimum-error "
                "sequential discrimination."
            ),
            overwrite=overwrite,
        )
        case_count = (
            len(MODELS)
            * len(K_VALUES)
            * len(M_VALUES)
            * REPLICATES_PER_CELL
        )
        print(f"Generated {dataset}: {case_count} cases in {target}")
        generated.append(dataset)
    return generated


def _parse_arguments() -> argparse.Namespace:
    """Parse benchmark selection and overwrite options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", choices=("all", *BENCHMARKS), nargs="?", default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Generate the requested compact benchmarks."""
    arguments = _parse_arguments()
    names = (
        list(BENCHMARKS)
        if arguments.benchmark == "all"
        else [arguments.benchmark]
    )
    generate_compact_benchmarks(names, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
