# Problem 2: sequential minimum-error discrimination

See paper §4.4 for methods, datasets and evaluation.

- `01_compact_analysis.ipynb`: Exhaustive, Greedy, VND + Beam, Uniform and Random.
- `02_extended_analysis.ipynb`: scalable methods on the extended dataset.
- `03_ablation_analysis.ipynb`: allocation and tree-order ablations; requires notebook 02 in the same mode.
- `final_figures.ipynb`: publication figures from completed full studies and ablation.
- [Problem 2 dual](problem-2-dual/README.md): target-error budgets (§4.5).

Use the [root setup](../README.md). Missing inputs can be generated from the repository root:

```bash
python problem-2-min-error-seq/scripts/generate_compact_benchmarks.py all
python problem-2-min-error-seq/scripts/generate_benchmarks.py all
```

In notebooks 01–02, `RUN_ANALYSIS=False` loads saved results; `True` evaluates cases. Choose `RUN_MODE="test"` or `"full"`; `RESUME=True` reuses matching checkpoints. Results: `results/{compact,extended,ablation}/{test,full}/`. Keep `storage_compatibility.json` with the project to preserve migrated cache identities.
