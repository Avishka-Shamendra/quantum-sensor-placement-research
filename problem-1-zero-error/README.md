# Minimum-Sensor Zero-Error Discrimination

This folder contains the zero-error sensor-allocation pipeline. It generates
one- and two-dimensional benchmark inputs, evaluates exact dynamic programming
and greedy approximation methods, and summarizes the resulting sensor budgets.

## Structure

```text
problem-1-zero-error/
├── 01_solution_method_analysis.ipynb
├── 02_minimum_sensor_trends.ipynb
├── 03_lower_bound_comparison.ipynb
├── final_figures.ipynb
├── src/
│   ├── optimal.py             # exact dynamic program
│   ├── lower_bound.py         # Section 3.2.1 Eq. (14) integer program
│   ├── sensor_budget_analysis.py # independent budget analyses from inputs
│   ├── run_config.py          # shared test/full selection and output paths
│   ├── test_lower_bound.py    # optional automated correctness checks
│   ├── greedy.py              # greedy gain-based approximation
│   ├── phase_data.py          # phase loading and threshold conversion
│   ├── field_models.py        # benchmark field families
│   ├── eval.py                # dataset evaluation
│   └── benchmark_analysis.py  # summaries, tables, and figures
├── scripts/
│   ├── generate_inputs.py
│   ├── generate_benchmarks.py
│   ├── evaluate_dataset.py
│   └── evaluate_benchmarks.py
├── inputs/                    # generated benchmark inputs
└── results/                   # generated tables and figures
```

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problem-1-zero-error/requirements.txt
```

## Run

Run the solution-method comparison from the command line:

```bash
python problem-1-zero-error/scripts/evaluate_benchmarks.py --mode test --require-existing
# Use --mode full for the complete solution-method comparison.
```

The three analysis notebooks run independently from the saved benchmark inputs,
in any order. Only `final_figures.ipynb` depends on their saved output tables:

```bash
jupyter lab problem-1-zero-error/01_solution_method_analysis.ipynb
jupyter lab problem-1-zero-error/02_minimum_sensor_trends.ipynb
jupyter lab problem-1-zero-error/03_lower_bound_comparison.ipynb
jupyter lab problem-1-zero-error/final_figures.ipynb
```

All three analysis notebooks have the same **Run configuration** cell:

```python
RUN_MODE = "test"  # Change to "full" for the complete benchmark.
```

Test mode selects 96 cases: both dimensions, all families and K values,
M=3 and 6, and replicate 1. Full mode selects all 11,520 cases. Each mode uses
the same solvers. Run all cells after choosing the mode. Notebook 01 uses 200
bootstrap resamples in test mode and 2,000 in full mode.

The first notebook runs the exact and greedy solution-method comparison. The
second recomputes the exact minimum-sensor trends directly from the raw inputs.
The third independently recomputes the exact sequential minimum and the global
integer lower bound on the same inputs, then plots both curves versus K for 1D
and 2D. The final notebook redraws the publication-ready figures from the three
analyses' **full-mode** saved tables. The command-line workflow above produces the
solution-method comparison only; run notebooks 02 and 03 for their analyses.

Each analysis requires existing benchmark inputs by default. If they are missing,
generate them with the command below; no other notebook is a prerequisite.

Generate benchmark inputs without evaluating algorithms:

```bash
python problem-1-zero-error/scripts/generate_benchmarks.py all
```

Evaluate one generated dataset:

```bash
python problem-1-zero-error/scripts/evaluate_dataset.py benchmark_1d
python problem-1-zero-error/scripts/evaluate_dataset.py benchmark_2d
```

## Benchmark Design

The standard benchmarks are `benchmark_1d` and `benchmark_2d`. Each contains
Gaussian, sinusoidal, and quadratic field families with `K = 3..10`,
`M = 3..10`, and 30 realizations per family, `K`, and `M` setting.

The main quality measure is

$$\rho = N_{\mathrm{greedy}} / N_{\min}^{\mathrm{seq}}.$$

Here $N_{\min}^{\mathrm{seq}}$ is the exact minimum sequential sensor budget
returned by the dynamic program, and $N_{\mathrm{greedy}}$ is the budget returned
by the greedy method. Existing CSV keys such as `N_optimal` and `mean_N_min_opt`
are retained for compatibility; displayed labels use $N_{\min}^{\mathrm{seq}}$.

## Separate scalability pilot

The original datasets and notebooks retain K,M=3..10. A separate input-only pilot
uses K={10,12,14,16,18,20}, M={5,10,15,20}, and three realizations per field-family
and K,M cell. It creates 216 cases in each of `inputs/scale_1d/` and
`inputs/scale_2d/`, preserving the main datasets:

```bash
python problem-1-zero-error/scripts/generate_benchmarks.py all --suite scale
```

Use `--replicates N` to choose another replication count when generating a new
pilot; replacing existing pilot files requires `--overwrite`. Generation does
not run any solvers. The standard comparison evaluator retains its K<=10 guard,
and notebooks 01-03 continue to select the main datasets. Larger exact runs need
explicit runtime and memory limits.

For presentation, the pilot grid needs only a 6-row by 4-column heatmap per
dimension, even though its maximum K and M are larger. Keep the original 8-by-8
heatmaps for the main benchmark; report scalability separately, including any
unsolved cases instead of treating them as exact results.

## Outputs

Generated files are written under `problem-1-zero-error/results/`:

```text
results/
├── methods/                  # notebook 01
│   ├── test/
│   └── full/
├── nmin/                     # notebook 02
│   ├── test/
│   └── full/
├── lb/                       # notebook 03
│   ├── test/
│   └── full/
└── final/                    # final_figures.ipynb
```

Each mode has `tables/` and `figures/` subfolders. Notebook 01 also saves detailed
per-dataset outputs in `1d/tables/` and `2d/tables/` inside the selected mode.
Tables are CSV files; figures are PDF and PNG files. Existing legacy output
folders are preserved; new runs use these short paths.

Notebook 03's selected mode folder includes:

- `tables/case_budgets.csv`: both optima, allocations, input hashes, solve times,
  lower-bound solver diagnostics, and per-instance gaps.
- `tables/nlb_vs_nmin_by_dimension_k.csv`: the main paired mean curves.
- `tables/nlb_vs_nmin_by_k_m.csv`: comparisons at fixed K and M.
- `figures/nlb_vs_nmin_by_k.{pdf,png}`: two panels, with both curves in each.

Test runs do not overwrite full results. Run all three notebooks with
`RUN_MODE = "full"` before running `final_figures.ipynb`, which reads only
`methods/full/`, `nmin/full/`, and `lb/full/`.

## Lower-bound interpretation and solver

In the supplied *Quantum Sensor Resource Allocation for Spatial Field
Discrimination* manuscript, Section 3.2.1, equation (14), defines

```text
N_LB_global = min sum_m n_m
subject to sum_m n_m * abs(r_ijm) >= pi for every i < j,
           n_m nonnegative integers.
```

`lower_bound.py` solves this ILP with SciPy/HiGHS using the original phase gaps.
It requests zero relative MIP gap, requires optimal termination and agreement
between the primal objective and dual bound, and validates the integer allocation
against every constraint to 1e-8 after normalization by pi. This is floating-point
numerical optimization, not a symbolic exact-arithmetic certificate. See the
[SciPy milp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

The comparison $N_{\min}^{\mathrm{seq}}$ is the exact **sequential** optimum in Section 3.3,
equation (15), computed with the existing dynamic program. Every feasible
sequential allocation satisfies the lower-bound constraints, so
$N_{\mathrm{LB}}^{\mathrm{global}} \leq N_{\min}^{\mathrm{seq}}$;
this is checked for every instance. The gap does not establish achievable global
savings: equation (14) alone does not ensure a common global input state exists.
In full mode, each plotted mean pools the same 720 cases per dimension and K (three families,
eight M values, 30 replicates). The fixed-K,M table retains location counts separately.

Optional automated correctness checks live in `src/test_lower_bound.py`; no
separate tests folder is needed. These checks are separate from notebook test
mode. Run them from the repository root:

```bash
python problem-1-zero-error/src/test_lower_bound.py -v
```
