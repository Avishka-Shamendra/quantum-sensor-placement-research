# Sequential vs. joint minimum-error discrimination

Matched comparison for paper §6: Problem 2 VND + Beam Search versus Problem 3
VND allocation with local equal-Dicke-sector superposition and PGM.
Both methods use identical inputs, priors, and actual sensor counts.

## Files

- `01_sequential_vs_joint_analysis.ipynb`: dataset preparation, evaluation, and reports.
- `comparison/config.py`: test/full settings and budget rounding.
- `comparison/dataset.py`: common inputs and joint lower bounds.
- `comparison/sources.py`: adapters to the original Problem 1–3 implementations.
- `comparison/runner.py`: resumable per-method evaluations.
- `comparison/reporting.py`: figure and field-family table.
- `data/<mode>/<fingerprint>/`: CSV inputs, seeds, parameters, and manifests.
- `results/<mode>/<fingerprint>/`: checkpoints, matched records, and exports.

## Run

Use the root README environment and retain all three original problem folders.
From the repository root:

```bash
python -m jupyterlab problem-2-3-comparison/01_sequential_vs_joint_analysis.ipynb
```

Run cells in order. The supplied notebook selects `RUN_MODE="full"`:
K,M=2–10, three families, two dimensions, 15 draws per combination
(7,290 before exclusions). `"test"` uses K,M=2,3 and two draws per combination.
The input cell generates missing datasets or loads matching inputs.
The evaluation cell automatically resumes completed method/budget checkpoints.
Changed source hashes or settings select separate caches.

Budgets are rounded fractions 0.1–0.9 of `min(N_LB_joint - 1, 50)`, with
halves rounded upward. Zero counts are dropped and duplicate counts evaluated once.
The sequential error includes off-pair outcomes; the joint error is the final
PGM error, not the allocation score.

Figure 1 averages matched errors at each target fraction, with pointwise
stratified bootstrap intervals. Table 1 averages unique budgets within each
instance, then instances within dimension/family. The win-rate cell counts each
unique instance/budget once, with tie tolerance 1e-9. Exports include PDF, PNG,
CSV, and Markdown.
