"""Generate the main benchmarks or a separate, wider scalability pilot."""

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


SCALABILITY = {
    name: {
        **design,
        "dataset": f"scale_{name}",
        "k_values": [10, 12, 14, 16, 18, 20],
        "m_values": [5, 10, 15, 20],
        "replicates": 3,
        "seed": 20260920 + 100_000 * int(design["dimension"]),
        "description": (
            f"{design['dimension']}-dimensional scalability pilot with wider K/M "
            "ranges; separate from the main benchmark and not evaluated at generation."
        ),
    }
    for name, design in BENCHMARKS.items()
}


def generate_benchmarks(names: list[str], overwrite: bool = False, *,
                        suite: str = "main", replicates: int | None = None) -> list[str]:
    """Generate selected benchmarks and return their dataset names."""
    if suite not in ("main", "scale"):
        raise ValueError("suite must be 'main' or 'scale'")
    if replicates is not None and (
        isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    designs = BENCHMARKS if suite == "main" else SCALABILITY
    if not names or any(name not in designs for name in names):
        raise ValueError("names must select '1d' and/or '2d'")
    # Check every destination before generating either dimension.
    for name in names:
        from generate_inputs import INPUT_ROOT
        target = INPUT_ROOT / str(designs[name]["dataset"])
        if not overwrite and ((target / "manifest.json").exists() or any(target.glob("case_*.csv"))):
            raise ValueError(f"Dataset already exists: {target}. Use --overwrite to replace it.")
    generated: list[str] = []
    for name in names:
        design = {**designs[name]}
        if replicates is not None:
            design["replicates"] = replicates
        print(f"Generating {design['dataset']}: K={design['k_values']}, "
              f"M={design['m_values']}, replicates={design['replicates']}", flush=True)
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
        print(f"Generated {name}: {case_count} cases in {target}", flush=True)
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
        "--suite", choices=("main", "scale"), default="main",
        help="main: existing K,M=3..10 design; scale: separate pilot up to 20.",
    )
    parser.add_argument(
        "--replicates", type=int,
        help="Realizations per family/K/M cell (default: main=30, scale=3).",
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
    generate_benchmarks(names, overwrite=arguments.overwrite,
                        suite=arguments.suite, replicates=arguments.replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
