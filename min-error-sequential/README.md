# Minimum-Error Sequential Discrimination

This directory contains the exact exhaustive-search pipeline for the
fixed-budget sequential discrimination problem, the greedy heuristic, and a
uniform-allocation baseline.

The two algorithm entry points are:

```python
from exhaustive_search import exhaustive_search
from greedy import greedy
from uniform import uniform

optimal_result = exhaustive_search(K, M, N, pijmn)
greedy_result = greedy(K, M, N, pijmn)
uniform_result = uniform(K, M, N, pijmn)
```

Generate the lite one- and two-dimensional input datasets from the repository
root:

```bash
python min-error-sequential/generate_benchmarks.py all
```

Run the exact algorithm:

```bash
python min-error-sequential/evaluate_benchmarks.py lite_benchmark_1d
python min-error-sequential/evaluate_benchmarks.py lite_benchmark_2d
```

For an interactive pilot with inline plots, open and run:

```bash
python -m pip install -r min-error-sequential/requirements.txt
jupyter lab min-error-sequential/evaluate_optimal.ipynb
```

The notebook evaluates one realization from every field-family, `K`, and `M`
cell by default. Set `RUN_ALL_REPLICATES = True` in its settings cell to run
all 3,000 cases per dataset. Notebook tables and figures are written under
`min-error-sequential/results/exhaustive_search_notebook/`.

To compare the exhaustive and greedy algorithms using paired cases, open:

```bash
jupyter lab min-error-sequential/compare_algorithms.ipynb
```

Its tables and candidate comparison figures are written under
`min-error-sequential/results/algorithm_comparison/`.

The generated inputs are stored under `inputs/`. Numerical results, optimal
allocations, pair locations, and decision trees are stored under `output/`.

The lite design uses Gaussian, sinusoidal, and quadratic fields in 1D and 2D,
with `K` in `{3, 4, 5}` and `M` in `{3, 4}`. Each dataset contains exactly
1,000 cases from each field family, for 3,000 cases in total. The six `K` and
`M` cells receive 166 or 167 cases per family. Each case uses up to four
distinct budgets below its zero-error minimum.

## Computational constraint

The lite benchmarks cap the sensor budget at 12 only to limit exhaustive-search
runtime. This is not a constraint of the sensor-allocation problem. The cap is
defined by `EXHAUSTIVE_SEARCH_BUDGET_CAP` in `generate_benchmarks.py`; custom
datasets can change it with `--budget-cap` or remove it with `--no-budget-cap`.

The CSV files contain only positive sensor counts. The algorithm assigns error
`0.5` when no stored count can be used. Generated cases use nonzero phase gaps
at stored locations, so this fallback value is not written to the CSV files.

For a custom or larger design, run:

```bash
python min-error-sequential/generate_inputs.py my_dataset \
  --dimension 2 --k-values 3 4 5 --m-values 3 4 5 \
  --replicates 3 --budget-cap 16 --samples-per-location 5
```

Increasing `K`, `M`, or the budget can increase the exhaustive-search runtime
rapidly.
