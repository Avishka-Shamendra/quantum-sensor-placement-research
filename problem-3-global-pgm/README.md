# Problem 3: multi-location joint discrimination

Five fixed input states evaluated with PGM, comparing Exhaustive, Greedy, VND,
Near-Uniform, and Random allocations (paper §5.5).

## Files

- `src/`: probe Gram matrices, PGM errors, allocation searches, and checkpoints.
- `scripts/generate_benchmarks.py`: current compact generation and extended-input reuse.
- `scripts/validate_benchmarks.py`: seed, CSV, budget, and joint-bound validation.
- `src/reporting/`: saved-result summaries and figure helpers.
- `scripts/`, `configs/`: input/evaluation utilities and earlier generation recipes;
  `generate_configured_benchmarks.py` retains the earlier recipes.
- `inputs/{compact,extended}/{1d,2d}/`: phase CSVs and manifests.
- `01_compact_analysis.ipynb`, `02_extended_analysis.ipynb`: the two studies.
- `final_figures.ipynb`: two publication figures and a compact runtime table.
- `results/`: saved records, tables, and PDF/PNG figures.

Compact uses K,M=3–5 and 50 replicates per family/combination: 2,700 instances.
Extended copies Problem 2's 4,320 extended instances byte-for-byte: K=3–10,
M=2–10, and 10 replicates per family/combination (2,160 per dimension). Both
use three field families and two dimensions, with max_m |r_ijm| ≥ 0.01 for
every field pair and c=1. Random draws one allocation per instance/budget,
shared across all five states. Extended gaps are relative to VND, not an optimum.

## Run

Use the root README environment. From the repository root:

```bash
python -m jupyterlab problem-3-global-pgm/01_compact_analysis.ipynb
```

Choose `RUN_MODE="test"` or `"full"`. The supplied study notebooks have
`RUN_ANALYSIS=True`, `RESUME=False`: running them starts fresh evaluations.
Set `RESUME=True` to reuse matching records, or `RUN_ANALYSIS=False` to load
matching saved results. Results use `results/{compact,extended}/{test,full}/`;
a fresh run overwrites that study/mode's records. The final-figures notebook
reads the two full result folders without running searches.

Regenerate and overwrite both extended datasets using the current Problem 2 inputs:

```bash
python problem-3-global-pgm/scripts/generate_benchmarks.py --studies extended --output problem-3-global-pgm/inputs --overwrite --workers 4
```

Omit `--studies extended` to regenerate both compact and extended studies.
Selected inputs are replaced only after generation succeeds. Generation has no
checkpoints. Rerun the corresponding study notebooks after replacing inputs.
Validation (includes joint-bound solves):

```bash
python problem-3-global-pgm/scripts/validate_benchmarks.py problem-3-global-pgm/inputs/compact
python problem-3-global-pgm/scripts/validate_benchmarks.py problem-3-global-pgm/inputs/extended
```
