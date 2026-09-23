"""Run actual-error comparisons on saved benchmark inputs."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from analysis import run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("compact", "extended"), default="compact")
    parser.add_argument("--mode", choices=("test", "full"), default="test")
    parser.add_argument("--exact-k-max", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_analysis(project=PROJECT, study=args.study, mode=args.mode, exact_k_max=args.exact_k_max,
                 time_limit=args.time_limit, resume=not args.fresh)


if __name__ == "__main__":
    main()
