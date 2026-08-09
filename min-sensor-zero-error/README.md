# Minimum-Sensor Zero-Error Discrimination

This directory contains the complete reproducible pipeline for comparing the
paper's **Exact Dynamic Programming** method with its **Greedy Gain-Based
Approximation**.

The research rationale, reporting plan, and limitations are documented in
[`PAPER_EVALUATION.md`](PAPER_EVALUATION.md).

## Recommended entry point

Create the environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r min-sensor-zero-error/requirements.txt
jupyter lab min-sensor-zero-error/evaluate_algorithms.ipynb
```

Run all notebook cells. The first run generates missing study datasets, runs
both methods on every case, computes uncertainty intervals, displays the
figures, and exports manuscript-ready PDF and PNG versions.

The same workflow can be run without Jupyter:

```bash
python min-sensor-zero-error/evaluate_paper_studies.py
```

For final runtime measurements, use at least five timing repetitions on an
otherwise idle machine:

```bash
python min-sensor-zero-error/evaluate_paper_studies.py --repeats 5
```

The quality results do not require repeats because both solvers are
deterministic; repeats only average timing noise.

## Paper-study design

The old 20-case `benchmark` is retained only as a quick smoke test. It is not
the dataset used to support paper claims.

| Dataset | Purpose | Factorial design | Cases |
|---|---|---:|---:|
| `study-primary` | Main quality and runtime evidence | 4 models × 8 `K` × 3 `M` × 30 realizations | 2,880 |
| `study-replication` | Independent-seed confirmation | Same factors × 10 new realizations | 960 |
| `study-sensitivity` | Paired phase-scale robustness | 4 models × 3 `K` × 2 `M` × 3 `c` × 15 realizations | 1,080 |

The primary and replication studies use `K = 3, …, 10`,
`M ∈ {3, 5, 8}`, and `c = 1`. The sensitivity study uses
`K ∈ {4, 7, 10}`, `M ∈ {3, 8}`, and `c ∈ {0.5, 1, 2}`. The same field
realization is reused across the three values of `c`, enabling paired
comparisons.

The four field families are Gaussian, sinusoidal, two-harmonic Fourier, and
cubic polynomial. Their formulas, sampled parameter ranges, seeds, and
realized parameters are recorded in source code and each dataset manifest.
Thirty independent realizations per primary factorial cell produce 360 cases
at each `K` and 720 per family. `sample_convergence.csv` checks the stability
of the pooled estimates from 5 through 30 realizations per cell.

Generate the three datasets without evaluating them:

```bash
python min-sensor-zero-error/generate_paper_studies.py all
```

Use `--overwrite` only when intentionally replacing an existing study. The
study definitions and fixed seeds are in `generate_paper_studies.py`.

The thousands of raw study CSVs and per-study solver JSON files are
deterministic intermediates and are intentionally ignored by Git. The notebook
recreates them from the committed designs and fixed seeds when they are absent.
Compact paper tables and figures in `output/paper-analysis/` are retained.

## Statistical reporting

The primary solution-quality measure is

```text
rho = N_greedy / N_optimal.
```

The analysis also reports exact-optimum recovery, percentage additional
sensors, the median, 95th percentile and maximum ratio, and runtimes. Mean
ratios receive reproducible 95% bootstrap intervals resampled within factorial
cells, preserving the balanced design; exact-match proportions receive 95%
Wilson intervals. The phase-scale analysis uses paired differences from
`c = 1` and stratified resampling.

The theorem gives

```text
rho <= 1 + ln(K(K - 1) / 2).
```

The figures compare the empirical excess with the theorem's permitted excess
using

```text
(rho - 1) / ln(K(K - 1) / 2).
```

This supports the limited statement that greedy is empirically closer to
optimal than the worst-case allowance on the stated synthetic distribution.
It does not establish a stronger theorem or a universal average-case
guarantee. A hypothesis test against the deterministic worst-case bound is not
appropriate.

## Outputs

`output/paper-analysis/` contains:

- `case_metrics.csv`: one transparent row per evaluated case.
- `group_summary.csv`: overall and pre-specified subgroup estimates with
  uncertainty intervals.
- `sample_convergence.csv`: estimate stability versus realizations per cell.
- `sensitivity_paired.csv`: paired phase-scale changes and bootstrap intervals.
- `paper_findings.md`: conservative manuscript-ready numerical wording.
- `primary_quality.*`: quality distributions, exact recovery, and gap to the
  worst-case allowance.
- `primary_quality_ecdf.*`: complete ratio distributions by field family.
- `primary_runtime_scaling.*`: median timings and interquartile bands.
- `phase_scale_sensitivity.*`: paired robustness analysis.
- `primary_sample_convergence.*`: sample-stability diagnostic.

Each figure is exported as vector PDF and 300-dpi PNG.

## Phase-gap convention

For fields `f_i` and `f_j` at location `x_m`, inputs follow the paper:

```text
r_ijm = Arg(exp(i c (f_i(x_m) - f_j(x_m)))) in (-pi, pi].
```

Only pairs with `i < j` are stored. The reverse orientation is derived as

```text
r_jim = principal_phase(-r_ijm).
```

Thus `r_jim = -r_ijm` modulo `2*pi`. At the branch cut, `pi` is the canonical
representation because the interval excludes `-pi`. The sensor threshold is

```text
N_ijm = ceil(pi / abs(r_ijm)).
```

A zero phase gap has no finite threshold at that location and is represented
internally by `None`. Every generated case ensures that each field pair has a
nonzero gap at least somewhere.

## Code layout

```text
min-sensor-zero-error/
├── optimal.py                  # Exact Dynamic Programming
├── greedy.py                   # Greedy Gain-Based Approximation
├── phase_data.py               # CSV validation and phase-to-threshold conversion
├── field_models.py             # transparent synthetic spatial-field families
├── generate_inputs.py          # general and factorial data generation
├── generate_paper_studies.py   # fixed paper-study definitions
├── eval.py                     # evaluates both methods on one dataset
├── paper_analysis.py           # statistics and publication figures
├── evaluate_paper_studies.py   # complete command-line workflow
├── evaluate_algorithms.ipynb   # documented interactive workflow
├── inputs/<dataset>/           # manifest and canonical phase-gap CSV files
└── output/                     # numerical results and paper artifacts
```

Reusable implementation belongs in `.py` files; the notebook should remain an
experiment runner. Importing `.ipynb` files into Python is intentionally not
part of the pipeline.

## Small custom datasets

For a quick custom experiment rather than the paper study:

```bash
python min-sensor-zero-error/generate_inputs.py experiment-02 \
  --cases 40 --k-min 3 --k-max 8 --m-min 3 --m-max 8 \
  --c 1.0 --seed 20260728
python min-sensor-zero-error/eval.py experiment-02
```

Dataset names remain neutral; field model, `K`, `M`, `c`, parameters, and seed
belong in `manifest.json`. Exact evaluation is intentionally limited to
`K <= 10` because the dynamic program is exponential.
