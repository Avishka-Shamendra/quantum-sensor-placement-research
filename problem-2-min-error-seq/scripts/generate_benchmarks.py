"""Generate the one- and two-dimensional benchmark datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HERE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from generate_inputs import generate_factorial_dataset


# Fixed benchmark parameters used for both spatial dimensions.
MODELS = ["gaussian", "sinusoidal", "quadratic"]
K_VALUES = list(range(3, 11))
M_VALUES = list(range(2, 11))
REPLICATES_PER_CELL = 30

# This cap limits algorithm runtime. It is not a constraint of the
# sensor-allocation problem and can be changed for other experiments.
BUDGET_CAP = 50

# Separate seeds make the 1D and 2D datasets reproducible and independent.
BENCHMARKS: dict[str, dict[str, object]] = {
    "1d": {
        "dataset": "benchmark_1d",
        "dimension": 1,
        "seed": 20260912,
    },
    "2d": {
        "dataset": "benchmark_2d",
        "dimension": 2,
        "seed": 20260913,
    },
}


def generate_benchmarks(names: list[str], overwrite: bool = False) -> list[str]:
    """Generate selected benchmarks and return their directory names."""
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
            budget_cap=BUDGET_CAP,
            seed=int(design["seed"]),
            description=(
                f"Expanded {name.upper()} benchmark for minimum-error "
                "sequential discrimination heuristic studies."
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
    """Generate the requested benchmarks."""
    arguments = _parse_arguments()
    names = list(BENCHMARKS) if arguments.benchmark == "all" else [arguments.benchmark]
    generate_benchmarks(names, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
