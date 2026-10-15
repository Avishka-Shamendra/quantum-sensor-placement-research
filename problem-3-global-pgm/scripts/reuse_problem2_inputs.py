"""Copy Problem 2 datasets unchanged; no field regeneration is required."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('compact_benchmark_1d', 'compact_benchmark_2d', 'benchmark_1d', 'benchmark_2d')


def convert(name, output_root, *, target=None):
    """Copy each case CSV and its manifest byte-for-byte into an empty target."""
    if name not in DATASETS:
        raise ValueError('unknown Problem 2 source dataset')
    source = ROOT.parent / 'problem-2-min-error-seq/inputs' / name
    target = Path(target) if target is not None else Path(output_root) / name
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'Refusing to replace {target}')
    original = json.loads((source/'manifest.json').read_text())
    paths = [source/case['file'] for case in original['cases']]
    if not all(p.is_file() for p in paths):
        raise ValueError('source dataset is incomplete')
    target.mkdir(parents=True, exist_ok=True)
    for path in [*paths, source/'manifest.json']:
        shutil.copyfile(path, target/path.name)
    print(f'{name}: {len(paths)} cases copied unchanged', flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'inputs')
    args = parser.parse_args()
    for name in DATASETS:
        convert(name, args.output_root)
