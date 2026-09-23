# Minimum-Error Sequential Discrimination

Compare sensor allocations and adaptive elimination trees using their actual
misclassification probability. This is an independent copy of the input datasets
and generators; `min-error-sequential` is unchanged. Previous results are not reused.

## Methods

- **Exhaustive:** enumerate allocations summing to N, then optimize every test pair,
  location, and branch subtree using actual outcome probabilities.
- **Greedy:** add one sensor at a time, choosing the trial allocation with the
  smallest average pair score. Each pair score is its minimum `P_ijmn` over all
  locations under that allocation. After allocation, build one tree by choosing
  the smallest `P_ijmn` at every node. The full tree is evaluated exactly once.
- **Heuristic:** near-uniform initialization, sensor-transfer moves of sizes 1, 2,
  and 3 using mean pairwise scores, followed by a beam-search tree (width 3).
  Scores guide construction only; the reported error comes from the final tree's
  actual outcome probabilities.
- **Uniform:** near-uniform allocation and a seeded random pair-location tree.
- **Random:** seeded random allocation and a seeded random pair-location tree.

All reported errors use the same physical evaluator, including outcomes when
the true field lies outside the tested pair. No construction score is reported
as a discrimination error.

## Model

Fields have equal priors. Each internal node tests two remaining fields using
the inherited probe construction and an equal-prior Helstrom measurement.
The zero eigenspace is split equally between outcomes. The outcome removes one
of the tested fields; the last field is declared.

Sensor counts are per-location capacities, reused after state preparation at
each node; they are not a cumulative count of measurements along a path. The
probe uses at most the pair's zero-error threshold. Off-pair branch
probabilities use the closed-form cases in Appendix D, equations (38), (41),
(43), (45), and (47). Exhaustive optimality is within this allocation, probe,
measurement, and elimination-tree model.

Each input CSV contains only `i,j,m,r_ijm`. The location column `m` identifies
the sensing location for each phase gap. Tested-pair and off-pair probabilities
are calculated from these signed phase gaps when a method runs.

## Run

From the repository root:

```bash
python -m pip install -r problem-2-min-error-seq/requirements.txt
jupyter lab problem-2-min-error-seq/01_compact_analysis.ipynb
```

- `01_compact_analysis.ipynb`: all five methods, including exhaustive search.
- `02_extended_analysis.ipynb`: the four scalable methods.
- `final_figures.ipynb`: figures and tables from saved full results only.

The first two notebooks run independently. Set `RUN_MODE` to `"test"` or
`"full"`. Test mode uses 180 cases per spatial dimension, balanced equally
across the three field families. It selects cases deterministically from each
dataset and uses every selected case's saved budgets.

Compact full runs use `EXACT_K_MAX = 4` by default. K=5 inputs are retained but
excluded unless this setting is increased. Actual exhaustive search considers
all sensing locations at every node and can be expensive. A 30-second limit
applies per method and budget; timeout records are not reported as optima.
Full runs may still take a long time.

```bash
python problem-2-min-error-seq/scripts/evaluate_benchmarks.py --study compact --mode test
python problem-2-min-error-seq/scripts/evaluate_benchmarks.py --study extended --mode test
```

Regenerate all four phase-only datasets with:

```bash
python problem-2-min-error-seq/scripts/generate_compact_benchmarks.py all --overwrite
python problem-2-min-error-seq/scripts/generate_benchmarks.py all --overwrite
```

`RESUME = True` reuses records only when inputs, case metadata, settings, and
source hashes match. Change settings or use `--fresh` for a new evaluation.
Completed records are appended immediately to `results/{compact,extended}/{test,full}/records.jsonl`.

## Inputs and outputs

The generated datasets use this design:

| Dataset | K | M | Replicates per family and cell | Cases per dimension |
| --- | --- | --- | --- | --- |
| Compact | 3–4 | 3–4 | 140 | 1,680 |
| Extended | 3–7 | 3–8 | 30 | 2,700 |

Both use Gaussian, sinusoidal, and quadratic families. Each case has up to four
budgets below its sequential zero-error minimum, capped at 12 sensors.
Generators are in `scripts/`; they reuse field generation and the zero-error
solver from `problem-1-zero-error` without modifying that folder.

Each family, dimension, K, and M cell uses an independent random calibration
sample to set the 70th percentile of the H-based conflict ratio. Generated
cases must pass basic phase feasibility, use at least two preferred sensing
locations, contain a competing location for at least 15% of pairs, and meet
their cell's fixed conflict threshold. Dataset selection never uses an
algorithm result. Manifests store the thresholds, cell acceptance rates, and
accepted-case structural diagnostics.

Saved results contain error probabilities, allocations, trees, runtime, and
completion status. `tables/error_summary.csv` and all comparison plots use the
same cases completed by every requested method. `tables/status.csv` gives the
status counts, while `tables/timeouts.csv` lists every timed-out dataset, case,
method, and budget together with its runtime and configured time limit. The
timeout file is written with headers even when no timeouts occur. Final figures
show error probability, runtime, and the compact exhaustive-error gap.

## Checks

```bash
python problem-2-min-error-seq/src/test_actual_error.py -q
```

The checks use tiny problems; full benchmark runs are separate.
