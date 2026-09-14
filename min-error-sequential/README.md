# Minimum-Error Sequential Discrimination

This folder contains the fixed-budget sequential discrimination pipeline. It
generates benchmark inputs, evaluates exact and heuristic methods, validates
surrogate errors against actual branch probabilities, and assembles final
figures.

## Structure

```text
min-error-sequential/
├── 01_compact_analysis.ipynb          # compact comparison and actual-error validation
├── 02_extended_analysis.ipynb         # extended comparison and heuristic ablation
├── 03_actual_optimality_gap.ipynb     # small actual-optimality gap study
├── 04_final_figures.ipynb             # final figure and table assembly
├── src/                               # algorithms, loaders, analysis, plotting
├── scripts/                           # input generation and command-line runners
├── docs/                              # method notes
├── inputs/                            # generated benchmark inputs
└── results/                           # generated tables, records, and figures
```

The main algorithms are `exhaustive_search.py`, `greedy.py`, `heuristic.py`,
`uniform.py`, and `random_baseline.py`. The heuristic starts from a
near-uniform allocation, runs deterministic VND moves with `L in {1, 2, 3}`,
and builds the final decision tree with beam width `B = 3`.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r min-error-sequential/requirements.txt
```

## Run

Generate inputs:

```bash
python min-error-sequential/scripts/generate_compact_benchmarks.py all
python min-error-sequential/scripts/generate_benchmarks.py all
```

Run the notebooks in order:

```bash
jupyter lab min-error-sequential/01_compact_analysis.ipynb
jupyter lab min-error-sequential/02_extended_analysis.ipynb
jupyter lab min-error-sequential/03_actual_optimality_gap.ipynb
jupyter lab min-error-sequential/04_final_figures.ipynb
```

The compact notebooks can run in test mode or full mode. Full extended
evaluation uses 2,700 cases per dimension and can take several hours.

## Outputs

Generated files are written under `min-error-sequential/results/`:

```text
results/
├── cmp_compact/
├── cmp_extended/
├── actual_error/
├── actual_optimal/
├── exact_optimal/
└── final_figures/
```

The input CSV files store positive sensor counts. When an algorithm requests an
unstored or unavailable count, the implementation uses error probability `0.5`
as the fallback.
