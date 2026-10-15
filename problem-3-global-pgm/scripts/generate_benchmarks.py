"""Generate all four dedicated Problem 3 datasets, preserving replaced inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from generate_inputs import ROOT, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace', action='store_true', help='Archive existing datasets before generating replacements')
    args = parser.parse_args()
    recipes = []
    for study in ('compact', 'extended'):
        for dimension in (1, 2):
            config = json.loads((ROOT/'configs'/f'{study}_{dimension}d.json').read_text())
            name = f'{"compact_" if study == "compact" else ""}benchmark_{dimension}d'
            recipes.append((config, ROOT/'inputs'/name))
    if not args.replace and any(target.exists() and any(target.iterdir()) for _,target in recipes):
        parser.error('Datasets already exist. Use --replace to archive and replace them.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for config, target in recipes:
        if target.exists() and any(target.iterdir()):
            archive = ROOT/'inputs'/'archive'/stamp/target.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            target.rename(archive)
            print(f'Archived {target.name} to {archive}')
        generate(config, target)
        print(f'Generated {target.name}', flush=True)


if __name__ == '__main__':
    main()
