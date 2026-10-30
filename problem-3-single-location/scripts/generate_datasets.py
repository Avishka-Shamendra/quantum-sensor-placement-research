"""Generate the 1D and 2D single-location datasets."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.dataset import DEFAULT_SEED, generate_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dimension', choices=('all', '1d', '2d'), nargs='?', default='all')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--min-phase-gap', type=float, default=0.01)
    parser.add_argument('--max-draws', type=int, default=100_000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    dimensions = (1, 2) if args.dimension == 'all' else (int(args.dimension[0]),)
    generate_datasets(dimensions, seed=args.seed, min_phase_gap=args.min_phase_gap,
                      max_draws=args.max_draws, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
