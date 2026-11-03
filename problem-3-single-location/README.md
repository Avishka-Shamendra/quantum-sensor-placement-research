# Problem 3: single-location optimization

Joint optimization of Dicke-sector weights and the POVM using an exact SDP.
The state and measurement variables satisfy `X_k >= 0`, `sum(X_k) = diag(p)`,
`p >= 0`, and `sum(p) = 1`. Reflection symmetry is not enforced.

## Files

- `scripts/generate_datasets.py`, `src/dataset.py`: Table 1 field generation.
- `inputs/{1d,2d}/{gaussian,sinusoidal,quadratic}/`: fields, phase gaps and manifests.
- `src/opt.py`, `sdp.py`: joint and fixed-state SDPs with feasibility and dual-gap checks.
- `src/states.py`: prescribed states and PGM errors.
- `src/analysis.py`, `reporting.py`: resumable evaluation and combined-dimension plots.
- `01_single_location_analysis.ipynb`: evaluation and saved-result reporting.
- `results/full/`: checkpoints, tables and figures from the exact SDP.

## Run

From the repository root:

```bash
python problem-3-single-location/scripts/generate_datasets.py all
python -m jupyterlab problem-3-single-location/01_single_location_analysis.ipynb
```

Existing inputs need not be regenerated. Generation uses 50 instances per dimension,
family and K=3,4,5, with c=1 and pairwise phase gaps at least 0.01. The same
900 instances are reused for N=2,4,6,8,10. Use `--overwrite` only to replace inputs.

Set `RUN_ANALYSIS=True` to evaluate missing cases; `False` loads saved exact-SDP
results. Checkpoints resume automatically. Automatic solver selection tries installed
SDP solvers; solutions failing numerical checks are not saved. SciPy remains required
for confidence intervals, not optimization.

Error curves average 1D and 2D equally, retaining the existing state comparisons
and three-family layouts. PGM uses the SDP-optimized weights. Reflection plots
pool points from both dimensions. Previous results are not used by the exact runner.

Checkpoint directories use dimension/family names, with input/source/settings fingerprints stored inside JSON records. Storage and plotting edits do not invalidate numerical work.
