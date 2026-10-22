# Problem 3: single-location optimization

Input-state and measurement comparisons for M=1 (paper §5.2).
K=3,4,5 and N=2,4,6,8,10, with 30 instances per field family and K:
270 field instances and 1,350 instance/budget combinations.

## Files

- `01_single_location_analysis.ipynb`: saved-result reporting and optional evaluation.
- `single_location/states.py`, `sdp.py`, `opt.py`: states, optimal measurements,
  and numerical state optimization.
- `single_location/analysis.py`: validation and resumable evaluation.
- `single_location/reporting.py`: tables and three publication figures.
- `imported/`: original CSV inputs, configurations, saved weights/results, and
  a checksum manifest. Older cases remain available but are excluded by the notebook.
- `results/`: measurements, optimized weights, checkpoints, tables, and PDF/PNG exports.
- `tests/`: small numerical checks.

## Run

Use the root README environment. From the repository root:

```bash
python -m jupyterlab problem-3-single-location/01_single_location_analysis.ipynb
```

The default `RUN_ANALYSIS=False` reads `results/measurements.csv` and saved weights
without optimization or SDP evaluation. Set `RUN_ANALYSIS=True` to validate inputs
and perform missing evaluations; matching checkpoints resume automatically.
Use one writer per checkpoint directory. The setup cell installs missing packages
into the active kernel if necessary.

These experiments use the supplied `imported/` datasets; there is no active
standalone dataset-generation command in this folder. Preserve the imported
files and checksum manifest together.

Optimized weights are numerical SLSQP solutions to the optimal-POVM objective,
not certified global optima. PGM uses the same optimized weights. Figures compare
optimal-POVM errors, PGM excess errors, and reflected Dicke-sector probabilities.
