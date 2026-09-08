"""Generate the two pre-specified synthetic benchmark datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HERE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from field_models import DEFAULT_MIN_BEST_PAIR_PHASE_GAP
from generate_inputs import generate_factorial_dataset


MODELS = ["gaussian", "sinusoidal", "quadratic"]

# Fixed designs make both benchmark datasets reproducible.
BENCHMARKS: dict[str, dict[str, object]] = {
    "1d": {
        "dataset": "benchmark_1d",
        "dimension": 1,
        "models": MODELS,
        "k_values": list(range(3, 11)),
        "m_values": list(range(3, 11)),
        "c_values": [1.0],
        "replicates": 30,
        "seed": 20260901,
        "min_best_pair_phase_gap": DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
        "description": (
            "One-dimensional synthetic benchmark on [0, 1] for comparing "
            "optimal and greedy zero-error sensor allocation."
        ),
    },
    "2d": {
        "dataset": "benchmark_2d",
        "dimension": 2,
        "models": MODELS,
        "k_values": list(range(3, 11)),
        "m_values": list(range(3, 11)),
        "c_values": [1.0],
        "replicates": 30,
        "seed": 20261017,
        "min_best_pair_phase_gap": DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
        "description": (
            "Two-dimensional synthetic benchmark on [0, 1]^2 for comparing "
            "optimal and greedy zero-error sensor allocation."
        ),
    },
}


def generate_benchmarks(names: list[str], overwrite: bool = False) -> list[str]:
    """Generate selected benchmarks and return their dataset names."""
    generated: list[str] = []
    for name in names:
        design = BENCHMARKS[name]
        target = generate_factorial_dataset(
            str(design["dataset"]),
            dimension=int(design["dimension"]),
            models=list(design["models"]),
            k_values=list(design["k_values"]),
            m_values=list(design["m_values"]),
            c_values=list(design["c_values"]),
            replicates=int(design["replicates"]),
            seed=int(design["seed"]),
            description=str(design["description"]),
            overwrite=overwrite,
            min_best_pair_phase_gap=float(design["min_best_pair_phase_gap"]),
        )
        case_count = (
            len(design["models"])
            * len(design["k_values"])
            * len(design["m_values"])
            * len(design["c_values"])
            * int(design["replicates"])
        )
        print(f"Generated {name}: {case_count} cases in {target}")
        generated.append(str(design["dataset"]))
    return generated


# Define command-line options for selecting and replacing benchmarks.
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the one- and two-dimensional benchmark datasets."
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        choices=("all", *BENCHMARKS),
        default="all",
        help="Benchmark to generate (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated benchmark.",
    )
    return parser.parse_args()


# Generate the requested benchmark datasets.
def main() -> int:
    arguments = _parse_arguments()
    names = list(BENCHMARKS) if arguments.benchmark == "all" else [arguments.benchmark]
    generate_benchmarks(names, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
