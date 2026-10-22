# Problem 3: multi-location joint discrimination

Five fixed input states evaluated with PGM, comparing Exhaustive, Greedy, VND,
Near-Uniform, and Random allocations (paper §5.5).

## Files

- `src/`: probe Gram matrices, PGM errors, allocation searches, and checkpoints.
- `benchmark_pipeline/generate.py`: current compact generation and extended-input reuse.
- `benchmark_pipeline/validate.py`: seed, CSV, budget, and joint-bound validation.
- `benchmark_pipeline/publication.py`, `final_figures.py`: reporting helpers.
- `scripts/`, `configs/`: input/evaluation utilities and earlier generation recipes;
  the current notebooks use the benchmark pipeline.
- `inputs/{compact,extended}/{1d,2d}/`: phase CSVs and manifests.
- `01_compact_analysis.ipynb`, `02_extended_analysis.ipynb`: the two studies.
- `final_figures.ipynb`: two publication figures and a compact runtime table.
- `results/`: saved records, tables, and PDF/PNG figures; `tests/`: small checks.

Compact uses K,M=3–5 and 50 replicates per family/combination: 2,700 instances.
Extended copies Problem 2's 4,320 extended instances. Both use three field
families and two dimensions. Random draws one allocation per instance/budget,
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

To regenerate inputs separately, first provide the Problem 2 extended datasets,
then choose an empty output directory:

```bash
python problem-3-global-pgm/benchmark_pipeline/generate.py --output /tmp/problem3-regenerated --workers 4
```

This writes `compact/` and `extended/` beneath that directory. Set each study
notebook's `DATASET_ROOT` to the corresponding directory when evaluating regenerated
inputs. The final-figures notebook expects the standard `inputs/` paths.
Generation has no checkpoints; interrupted generation needs a new empty directory.
Validation (includes joint-bound solves):

```bash
python problem-3-global-pgm/benchmark_pipeline/validate.py problem-3-global-pgm/inputs/compact
python problem-3-global-pgm/benchmark_pipeline/validate.py problem-3-global-pgm/inputs/extended
```
