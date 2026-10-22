# Problem 2 dual: sensor budgets for a target error

Greedy and VND + Beam Search budget searches for paper §4.7. Reported counts
are verified feasible solutions, not certified global minima.

## Files

- `01_extended_analysis.ipynb`: configuration, resumable searches, and reports.
- `dual_search.py`: binary budget search with a verified perfect-discrimination fallback.
- `greedy_comparison.py`: the corresponding independent Greedy search.
- `budget_limited.py`: shared instance cutoff and cache reuse.
- `reporting.py`: paired summaries, figure, and tables.
- `results/`: per-instance checkpoints, saved solutions, and exports.
- `tests/`: small cutoff and cache checks.

## Run

Use the root README environment and the existing Problem 2 extended inputs.
No separate dataset generation is needed. From the repository root:

```bash
python -m jupyterlab problem-2-min-error-seq/problem-2-dual/01_extended_analysis.ipynb
```

`RUN_ANALYSIS=True` runs missing work; `RESUME=True` reuses matching steps.
Choose `RUN_MODE="test"` or `"full"`. Targets are 0.1–0.5.
`MAX_SENSOR_BOUND=5000` retains 2,155 1D and 2,134 2D full-study instances;
exclusions are saved in the coverage outputs.

Each target starts with budget bounds 0 and the sequential perfect-discrimination
minimum. Actual tree error determines each binary-search update.
For reports only, set `RUN_ANALYSIS=False` and point `SAVED_RESULTS` to a
completed run directory. Limited comparisons save under `results/comparison_limited/`.
The figure shows median feasible counts and interquartile bands.
