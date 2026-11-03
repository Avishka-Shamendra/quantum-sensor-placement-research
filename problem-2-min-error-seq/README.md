# Problem 2: sequential minimum-error discrimination

Allocation and decision-tree comparisons using complete discrimination errors,
including off-pair outcomes (paper §4.4).

## Files

- `src/`: physical error evaluation, Exhaustive, Greedy, VND + Beam Search,
  Uniform/Random baselines, checkpointing, and plot helpers.
- `scripts/`: seeded benchmark generation and evaluation entry points.
- `inputs/`: signed phase-gap CSVs and instance manifests.
- `01_compact_analysis.ipynb`: all five methods, including Exhaustive.
- `02_extended_analysis.ipynb`: the four scalable methods.
- `03_ablation_analysis.ipynb`: allocation and test-order ablations after notebook 02.
- `src/ablation.py`: matched ablations, checkpoints, and the combined 1D/2D figure.
- `final_figures.ipynb`: publication figures from saved full results.
- `results/`: records, tables, figures, timeouts, and supplementary studies.
- [problem-2-dual/](problem-2-dual/README.md): target-error budget searches (§4.5).

Both dimensions use Gaussian, sinusoidal, and quadratic fields. Compact:
K,M=3–4, 50 replicates per combination, 600 instances per dimension.
Extended: K=3–10, M=2–10, 10 replicates, 2,160 instances per dimension.
Every field pair must have at least one location with `abs(r_ijm) >= 0.01`,
using the same validity threshold as Problem 1. Rejected draws are resampled.
Up to four distinct budgets are evaluated per instance.
Notebooks 01–02 also save VND iteration distributions.
These require newly recorded iteration diagnostics; older results cannot supply them.

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
Evaluations have no time limit by default. Run notebook 03 after notebook 02,
using the same mode. It reuses Greedy/VND allocations and VND + Beam Search errors,
then evaluates Near-Uniform + Beam Search, Greedy + Beam Search, and VND + Random
Tree. Each instance-budget pair uses one seeded random tree. Results and resumable
checkpoints are saved under `results/ablation/`; no new inputs are generated.

Final figures require both full studies and the full ablation. Restart the kernel
when switching notebooks. Checks:
`python problem-2-min-error-seq/src/test_actual_error.py -q`.
