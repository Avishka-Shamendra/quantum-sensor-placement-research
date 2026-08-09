"""Generate the pre-specified datasets used for paper evaluation.

The three studies separate the primary quality claim, an independent-seed
replication, and sensitivity to the physical phase-scale constant ``c``.
"""

from __future__ import annotations

import argparse

from field_models import FIELD_MODELS
from generate_inputs import generate_factorial_dataset


MODELS = list(FIELD_MODELS)
STUDIES = {
    "primary": {
        "dataset": "study-primary",
        "models": MODELS,
        "k_values": list(range(3, 11)),
        "m_values": [3, 5, 8],
        "c_values": [1.0],
        "replicates": 30,
        "seed": 20260901,
        "description": (
            "Primary balanced factorial study of greedy solution quality and "
            "runtime relative to the exact optimum."
        ),
    },
    "replication": {
        "dataset": "study-replication",
        "models": MODELS,
        "k_values": list(range(3, 11)),
        "m_values": [3, 5, 8],
        "c_values": [1.0],
        "replicates": 10,
        "seed": 20261017,
        "description": (
            "Independent-seed confirmation of the primary factorial study."
        ),
    },
    "sensitivity": {
        "dataset": "study-sensitivity",
        "models": MODELS,
        "k_values": [4, 7, 10],
        "m_values": [3, 8],
        "c_values": [0.5, 1.0, 2.0],
        "replicates": 15,
        "seed": 20261103,
        "description": (
            "Paired sensitivity study for the phase-scale constant c at low, "
            "middle, and high problem sizes."
        ),
    },
}


def generate_studies(names: list[str], overwrite: bool = False) -> list[str]:
    """Generate selected pre-specified studies and return their dataset names."""
    generated: list[str] = []
    for name in names:
        design = STUDIES[name]
        target = generate_factorial_dataset(
            str(design["dataset"]),
            models=list(design["models"]),
            k_values=list(design["k_values"]),
            m_values=list(design["m_values"]),
            c_values=list(design["c_values"]),
            replicates=int(design["replicates"]),
            seed=int(design["seed"]),
            description=str(design["description"]),
            overwrite=overwrite,
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


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper evaluation studies.")
    parser.add_argument(
        "study",
        nargs="?",
        choices=("all", *STUDIES),
        default="all",
        help="Study to generate (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated study.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    names = list(STUDIES) if arguments.study == "all" else [arguments.study]
    generate_studies(names, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
