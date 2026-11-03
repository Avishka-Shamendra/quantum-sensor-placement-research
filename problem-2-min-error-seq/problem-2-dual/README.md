# Problem 2 dual: target-error sensor budgets

See paper §4.5. VND + Beam and Greedy search for verified feasible budgets; the returned counts are not certified global minima.

Use the [root setup](../../README.md) and open `01_extended_analysis.ipynb`. Existing Problem 2 extended inputs are required.

- `RUN_MODE`: `"full"` or `"compact"` (`"test"` is an alias).
- `METHODS`: select VND + Beam, Greedy or both.
- `RUN_ANALYSIS=False`: report from `SAVED_RESULTS`; `True` runs searches.
- `RESUME=True`: retain matching completed searches; `MAX_SENSOR_BOUND` controls inclusion.

Outputs: `results/{full,compact}/`; shared checkpoints: `results/cache/`. Retain `compatible_signatures.json` with the cache. Curves show medians with interquartile bands; paired summaries use completed evaluations for both methods.
