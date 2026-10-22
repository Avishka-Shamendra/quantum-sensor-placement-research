# Problem 2: sequential minimum-error discrimination

Allocation and decision-tree comparisons using complete discrimination errors,
including off-pair outcomes (paper §4.6).

## Files

- `src/`: physical error evaluation, Exhaustive, Greedy, VND + Beam Search,
  Uniform/Random baselines, checkpointing, and plot helpers.
- `scripts/`: seeded benchmark generation and evaluation entry points.
- `inputs/`: signed phase-gap CSVs and instance manifests.
- `01_compact_analysis.ipynb`: all five methods, including Exhaustive.
- `02_extended_analysis.ipynb`: the four scalable methods.
- `final_figures.ipynb`: publication figures from saved full results.
- `results/`: records, tables, figures, timeouts, and supplementary studies.
- [problem-2-dual/](problem-2-dual/README.md): target-error budget searches (§4.7).

Both dimensions use Gaussian, sinusoidal, and quadratic fields. Compact:
K,M=3–4, 50 replicates per combination, 600 instances per dimension.
Extended: K=3–10, M=2–10, 10 replicates, 2,160 instances per dimension.
Up to four distinct budgets are evaluated per instance.

## Run

Use the root README environment; keep `problem-1-zero-error/` alongside this folder.
To generate missing inputs from the repository root:

```bash
python problem-2-min-error-seq/scripts/generate_compact_benchmarks.py all
python problem-2-min-error-seq/scripts/generate_benchmarks.py all
python -m jupyterlab problem-2-min-error-seq/01_compact_analysis.ipynb
```

Generators support `--overwrite` for deliberate replacement. In notebooks 01–02,
`RUN_ANALYSIS=False` loads saved results; `True` runs missing evaluations.
Choose `RUN_MODE="test"` or `"full"`; `RESUME=True` reuses matching records.
Test samples contain 90 compact or 180 extended cases per dimension.

Alternatively:

```bash
python problem-2-min-error-seq/scripts/evaluate_benchmarks.py --study compact --mode full
python problem-2-min-error-seq/scripts/evaluate_benchmarks.py --study extended --mode full
```

Records are saved in `results/{compact,extended}/{test,full}/records.jsonl`.
The default per-method/budget timeout is 30 seconds; incomplete cases are
reported separately. Final figures require both full studies. Checks:
`python problem-2-min-error-seq/src/test_actual_error.py -q`.
