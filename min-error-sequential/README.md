# Minimum-Error Sequential Discrimination

This directory contains the fixed-budget sequential discrimination pipeline:
exact exhaustive search for compact cases, Greedy and Tree-aware heuristic
methods, and Uniform and Random baselines.

Reusable algorithm and analysis modules live in `src/`. Add that directory to
`PYTHONPATH` before importing the algorithms outside the notebooks:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("min-error-sequential/src").resolve()))

from exhaustive_search import exhaustive_search
from greedy import greedy
from heuristic import heuristic
from random_baseline import random_baseline
from uniform import uniform

optimal_result = exhaustive_search(K, M, N, pijmn)
greedy_result = greedy(K, M, N, pijmn)
tree_aware_result = heuristic(K, M, N, pijmn)
uniform_result = uniform(K, M, N, pijmn)
random_result = random_baseline(K, M, N, pijmn)
```

The Tree-aware heuristic accepts optional `beam_width`, `max_iterations`,
`improvement_threshold`, `initialization`, `random_seed`,
`include_greedy_start`, and `move_shortlist_size` keyword arguments. By
default, it starts from the greedy allocation, ranks all one-sensor transfers
using the current tree's pair weights, rebuilds the beam-search tree only for
the top-ranked shortlist, and accepts the best full-tree improvement. Set
`initialization="near_uniform"` or `initialization="random"` to use a
non-greedy starting allocation; `include_greedy_start=True` can add the greedy
allocation as an additional start for those modes.

The uniform baseline uses every sensor. It assigns `N // M` sensors to every
location and randomly selects distinct locations to receive the remainder.
Its optional `random_seed` argument makes this near-uniform allocation
reproducible.

The Random baseline places every sensor at a uniformly sampled location and
selects a random valid field pair at each decision-tree node. It still uses the
best sensing location supported by that random allocation for each pair. Its
optional `random_seed` argument makes the strategy reproducible.

Generate the expanded one- and two-dimensional input datasets from the repository
root:

```bash
python min-error-sequential/scripts/generate_benchmarks.py all
```

Generate the compact one- and two-dimensional input datasets:

```bash
python min-error-sequential/scripts/generate_compact_benchmarks.py all
```

Run the exact algorithm on the compact datasets:

```bash
python min-error-sequential/scripts/evaluate_benchmarks.py compact_benchmark_1d
python min-error-sequential/scripts/evaluate_benchmarks.py compact_benchmark_2d
```

## Notebooks

The submission-facing notebooks are kept directly in this folder:

- `01_compact_exact_comparison.ipynb`: compares Exhaustive, Greedy,
  Tree-aware heuristic, Uniform, and Random on the compact Problem 2 data using
  surrogate final error.
- `02_actual_error_validation.ipynb`: reloads saved allocations and decision
  trees and reevaluates them with actual Helstrom branch probabilities.
- `03_scalable_algorithm_comparison.ipynb`: compares the scalable methods on the
  expanded data using surrogate final error.
- `04_heuristic_diagnostics.ipynb`: studies beam-width sensitivity for the
  Tree-aware heuristic.

To install notebook dependencies:

```bash
python -m pip install -r min-error-sequential/requirements.txt
```

To compare all five algorithms against the exact optimum on the compact datasets,
open:

```bash
jupyter lab min-error-sequential/01_compact_exact_comparison.ipynb
```

Its test mode uses 180 cases per dataset. Results are written under
`min-error-sequential/results/algorithm_comparison_compact/`.

To compare Greedy, Heuristic, Uniform, and Random on the expanded datasets, open:

```bash
jupyter lab min-error-sequential/03_scalable_algorithm_comparison.ipynb
```

Its test mode uses 100 balanced cases per dataset, while full mode uses all
2,700 cases.
Results are written under `min-error-sequential/results/algorithm_comparison/`.

To evaluate whether the current `0.5/0.5` outside-pair surrogate tracks the
actual quantum measurement probabilities, open:

```bash
jupyter lab min-error-sequential/02_actual_error_validation.ipynb
```

This notebook reads the saved compact paired-comparison records, recomputes
Exhaustive, Greedy, Tree-aware, Uniform, and Random final errors using the
physical pairwise probe and symmetric Helstrom POVM, and writes
actual-vs-surrogate tables and figures under
`min-error-sequential/results/actual_error_evaluation/`.

To study how beam width affects the tree-aware heuristic, open:

```bash
jupyter lab min-error-sequential/04_heuristic_diagnostics.ipynb
```

The default study compares beam widths 1, 3, and 5 on `benchmark_1d` and
`benchmark_2d`. Its tables and side-by-side 1D/2D figure are written under
`min-error-sequential/results/heuristic_beam_width/`.

The sequential generated inputs are stored under `min-error-sequential/inputs/`.
All numerical results, allocations, pair locations, decision trees, and
figures remain under `min-error-sequential/results/`.

The main comparison follows the same layout as the zero-error folder:

```text
results/
├── algorithm_comparison_compact/
│   ├── compact_benchmark_1d_results_<test|full>/tables/
│   ├── compact_benchmark_2d_results_<test|full>/tables/
│   └── results_summary_compact_benchmark_1d_compact_benchmark_2d_<test|full>/figures/
├── algorithm_comparison/
│   ├── benchmark_1d_results_<test|full>/tables/
│   ├── benchmark_2d_results_<test|full>/tables/
│   └── results_summary_benchmark_1d_benchmark_2d_<test|full>/figures/
└── optimal_evaluation/       # exhaustive-search-only results
```

The `_test` suffix identifies a notebook pilot, while `_full` identifies a run
over every selected case. Optimal-only notebook outputs follow the same
`<dataset>_results_<test|full>` rule.

The compact design uses `K=3,4,5` and `M=3,4`. Each of its 18 field-family,
`K`, and `M` cells contains 140 cases, giving 2,520 cases per dimension.

The expanded design uses Gaussian, sinusoidal, and quadratic fields in 1D and
2D, with `K` ranging from 3 through 7 and `M` ranging from 3 through 8. Every
field-family, `K`, and `M` cell has 30 independent cases. Each dataset therefore
contains 2,700 cases: 3 field families times 5 `K` values times 6 `M` values times 30
replicates. Each case uses up to four distinct budgets below its zero-error
minimum.

## Computational constraint

The benchmarks cap the sensor budget at 12 only to limit algorithm runtime.
This is not a constraint of the sensor-allocation problem. The cap does not
make exhaustive search practical for the largest `K` values, so the expanded
datasets are intended primarily for heuristic studies. The compact datasets
remain available for exact-optimal comparisons. The cap is defined by
`EXHAUSTIVE_SEARCH_BUDGET_CAP` in the relevant benchmark generator; custom
datasets can change it with `--budget-cap` or remove it with `--no-budget-cap`.

The beam-width notebook uses three allocation iterations by default. Representative 2D,
`M=8`, `N=12` timings showed that `K=8` was several times slower than `K=7`,
so `K=8` is excluded from the full factorial run. A timing sample across one
replicate of every retained cell projected about 11 hours for both complete
datasets on the development machine.

The CSV files contain only positive sensor counts. The algorithm assigns error
`0.5` when no stored count can be used. Generated cases use nonzero phase gaps
at stored locations, so this fallback value is not written to the CSV files.

For a custom or larger design, run:

```bash
python min-error-sequential/scripts/generate_inputs.py my_dataset \
  --dimension 2 --k-values 3 4 5 --m-values 3 4 5 \
  --replicates 3 --budget-cap 16 --samples-per-location 5
```

Increasing `K`, `M`, or the budget can increase the exhaustive-search runtime
rapidly.
