# Problem 3: joint minimum-error discrimination

See paper §§5.2–5.3 for input states, allocation methods and evaluation. Greedy minimizes PGM error for each state; VND uses the phase-separation score.

- `01_compact_analysis.ipynb`: compact comparison against Exhaustive.
- `02_extended_analysis.ipynb`: extended comparison, with Greedy as the gap reference.
- `03_greedy_diagnostics.ipynb`: saved compact allocation/error diagnostics.
- `final_figures.ipynb`: publication figures and runtime/gap summaries from saved full results.

Use the [root setup](../README.md). Missing inputs can be prepared with `python problem-3-global-pgm/scripts/generate_benchmarks.py --output problem-3-global-pgm/inputs` from the repository root; extended inputs reuse Problem 2's dataset.

Choose `RUN_MODE="test"` or `"full"`. `RUN_ANALYSIS=False` loads matching saved results; `True` runs evaluations. `RESUME=True` reuses matching records; a fresh run overwrites the selected study/mode. Outputs: `results/{compact,extended}/{test,full}/` and `results/final_figures/`.
