"""Generate compact global-discrimination benchmarks in 1D and 2D."""

from __future__ import annotations

import argparse

from generate_inputs import generate_factorial_dataset


MODELS = ["gaussian", "sinusoidal", "quadratic"]
K_VALUES = [3, 4, 5]
M_VALUES = [3, 4]
REPLICATES_PER_CELL = 140
BENCHMARKS: dict[str, dict[str, object]] = {
    "1d": {
        "dataset": "compact_benchmark_1d",
        "dimension": 1,
        "seed": 20260901,
    },
    "2d": {
        "dataset": "compact_benchmark_2d",
        "dimension": 2,
        "seed": 20260902,
    },
}


def generate_compact_benchmarks(
    names: list[str],
    overwrite: bool = False,
) -> list[str]:
    """Generate selected independent compact global benchmarks."""
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
            phase_scale=1.0,
            seed=int(design["seed"]),
            description=(
                f"Independent compact {name.upper()} benchmark for "
                "exact minimum-error global discrimination."
            ),
            overwrite=overwrite,
        )
        print(f"Generated {dataset}: 2,520 cases in {target}")
        generated.append(dataset)
    return generated


def main() -> int:
    """Generate requested compact benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", choices=("all", *BENCHMARKS), nargs="?", default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    names = (
        list(BENCHMARKS)
        if arguments.benchmark == "all"
        else [arguments.benchmark]
    )
    generate_compact_benchmarks(names, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
