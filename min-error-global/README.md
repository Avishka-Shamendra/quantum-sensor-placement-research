# Minimum-Error Global Discrimination

This directory owns the inputs, algorithms, and comparison workflow for the
fixed-budget global-discrimination problem. The global and sequential problems
use independent generated field instances.

## Objective And Inputs

Each global CSV stores canonical unwrapped phase differences:

```text
i,j,m,delta_ijm
```

Here `delta_ijm = c*(f_i(x_m)-f_j(x_m))`; it is not reduced modulo `2*pi`.
The loader verifies that all field triples are mutually consistent, which is
required to construct one positive-semidefinite output-state Gram matrix.

All three algorithms minimize the actual equal-prior Pretty Good Measurement
(PGM) error obtained from that Gram matrix. The analytical Section V upper bound
is still returned as a secondary reporting metric.

```python
from algorithms.greedy import greedy_global_discrimination
from algorithms.optimal import optimal_global_discrimination
from algorithms.vnd import vnd_global_discrimination
from input_data import load_delta_input

K, M, delta_ijm = load_delta_input(
    "min-error-global/inputs/compact_benchmark_1d/case_00001.csv"
)
result = optimal_global_discrimination(
    K,
    M,
    N=5,
    delta_ijm=delta_ijm,
    state_model="product",
)
print(result.allocation, result.error_probability, result.error_upper_bound)
```

Every algorithm takes `K`, `M`, an absolute sensor budget `N`, `delta_ijm`, and
one state model: `product`, `local_ghz`, or `global_ghz`.

VND starts from a deterministic near-uniform allocation. It searches exact
two-location (V1) and three-location (V2) redistributions in lexicographic
order, with candidates ordered by local L1 distance and then lexicographically.
The first actual-error reduction of at least `1e-3` is accepted, after which the
search restarts from V1. Full-allocation objective evaluations are cached.

## Independent Benchmarks

The generators reuse the Gaussian, sinusoidal, and quadratic field functions
and parameter ranges from Tables I and II, but use global-specific seeds.

| Dataset | K | M | Replicates per family/K/M | Cases |
| --- | --- | --- | ---: | ---: |
| `compact_benchmark_1d` | 3-5 | 3-4 | 140 | 2,520 |
| `compact_benchmark_2d` | 3-5 | 3-4 | 140 | 2,520 |
| `benchmark_1d` | 3-7 | 3-8 | 30 | 2,700 |
| `benchmark_2d` | 3-7 | 3-8 | 30 | 2,700 |

Generate the four datasets from the repository root:

```bash
python min-error-global/generate_compact_benchmarks.py all
python min-error-global/generate_benchmarks.py all
```

The compact datasets are intended for exact comparisons. The extended datasets
support scalable heuristic studies because exhaustive allocation search becomes
expensive as `M` and `N` increase.

## Comparison Notebook

Install dependencies and open the notebook:

```bash
python -m pip install -r min-error-global/requirements.txt
jupyter lab min-error-global/compare_optimal_greedy.ipynb
```

The Boolean `RUN_ALL_CASES` selects 100 reproducibly sampled cases per compact
dataset (`False`) or all 2,520 cases (`True`). The default budgets are `N=1..10`.
The notebook produces absolute upper-bound and actual-error plots for all three
methods, plus Optimal-relative differences for both metrics.

Results are written under:

```text
min-error-global/results/actual_pgm_optimal_greedy_vnd/
```
