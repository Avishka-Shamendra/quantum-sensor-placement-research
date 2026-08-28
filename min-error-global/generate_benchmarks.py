"""Generate extended global-discrimination benchmarks in 1D and 2D."""

from __future__ import annotations

import argparse

from generate_inputs import generate_factorial_dataset


MODELS = ["gaussian", "sinusoidal", "quadratic"]
K_VALUES = list(range(3, 8))
M_VALUES = list(range(3, 9))
REPLICATES_PER_CELL = 30
BENCHMARKS: dict[str, dict[str, object]] = {
    "1d": {
        "dataset": "benchmark_1d",
        "dimension": 1,
        "seed": 20260903,
    },
    "2d": {
        "dataset": "benchmark_2d",
        "dimension": 2,
        "seed": 20260904,
    },
}


def generate_benchmarks(names: list[str], overwrite: bool = False) -> list[str]:
    """Generate selected independent extended global benchmarks."""
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
                f"Independent extended {name.upper()} benchmark for "
                "minimum-error global discrimination."
            ),
            overwrite=overwrite,
        )
        print(f"Generated {dataset}: 2,700 cases in {target}")
        generated.append(dataset)
    return generated


def main() -> int:
    """Generate requested extended benchmarks."""
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
    generate_benchmarks(names, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
