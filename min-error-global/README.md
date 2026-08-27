# Minimum-Error Global Discrimination

This directory implements the exhaustive and greedy fixed-budget allocation
algorithms from Section V of the paper.

Only the two algorithm entry points live under `algorithms/`:

```python
from algorithms.greedy import greedy_global_discrimination
from algorithms.optimal import optimal_global_discrimination
```

Both functions take `K`, `M`, a fixed sensor budget `N`, canonical phase gaps
`rijm`, and one input-state model: `product`, `local_ghz`, or `global_ghz`.

## Shared Inputs

Global and sequential discrimination use the same four sparse benchmark data
sets under `min-error-datasets/inputs/`:

- `compact_benchmark_1d`: `K=3..5`, `M=3..4`, 2,520 cases
- `compact_benchmark_2d`: `K=3..5`, `M=3..4`, 2,520 cases
- `benchmark_1d`: `K=3..7`, `M=3..8`, 2,700 cases
- `benchmark_2d`: `K=3..7`, `M=3..8`, 2,700 cases

Each CSV includes both `r_ijm` and the pairwise error values required by the
sequential problem. The global algorithms use `r_ijm` and ignore the pairwise
error table:

```python
import sys

sys.path.insert(0, "min-error-datasets")
sys.path.insert(0, "min-error-global")

from algorithms.greedy import greedy_global_discrimination
from input_data import load_sparse_input

K, M, rijm, _ = load_sparse_input(
    "min-error-datasets/inputs/compact_benchmark_1d/case_00001.csv"
)
result = greedy_global_discrimination(
    K, M, N=5, rijm=rijm, state_model="product"
)
```

The sensor budgets already recorded in each manifest can be reused for a
paired comparison with the sequential algorithms.
