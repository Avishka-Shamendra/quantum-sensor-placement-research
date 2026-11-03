# Problem 1: perfect discrimination

Sequential dynamic programming and Greedy allocation, with a joint
perfect-discrimination sensor lower bound (paper §§3.4–3.5).

## Files

- `src/`: solvers, shared field models, input readers, and analysis helpers.
- `scripts/`: dataset generation and command-line evaluation.
- `inputs/`: phase-gap CSVs and manifests; 19,800 cases per dimension,
  K=2–12, M=1–12, three field families, 50 replicates per combination.
- `01_solution_method_analysis.ipynb`: DP/Greedy errors in sensor count,
  runtime, and allocation agreement; saves `results/methods/`.
- `02_minimum_sensor_trends.ipynb`: sequential budgets; saves `results/nmin/`.
- `03_global_seq_comparison.ipynb`: joint lower bound and sequential budgets;
  saves `results/global/`.
- `final_figures.ipynb`: publication figures from saved full results.
- `sanity-check/`: sparse-state overlap checks and their notebook.
- `results/`: saved tables, figures, and runtime studies.

## Run

Commands below assume the repository-root environment described in the root README.
Generate missing benchmarks, then open the analysis notebook:

```bash
python problem-1-zero-error/scripts/generate_benchmarks.py all
python -m jupyterlab problem-1-zero-error/01_solution_method_analysis.ipynb
```

Existing datasets are protected; `--overwrite` explicitly replaces them.
Notebooks 01–03 use `RUN_MODE="test"` (132 cases) or `"full"` (39,600 cases).
They can run independently. In 02–03, `REUSE_SAVED_RESULTS=True` reuses matching
DP results; missing results still require computation. Run all three in full
mode before rebuilding `final_figures.ipynb`.

The command-line method comparison is:

```bash
python problem-1-zero-error/scripts/evaluate_benchmarks.py --mode full --require-existing
```

The overlap notebook uses `RUN_SCOPE="test"` or `"full"` and writes
`sanity-check/outputs/`. Solver checks: `python problem-1-zero-error/src/test_global_minimum.py -q`.
