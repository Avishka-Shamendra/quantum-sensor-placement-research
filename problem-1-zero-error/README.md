# Zero-Error Sensor Allocation

Compare sequential DP, sequential Greedy, and a global integer-program minimum
on one- and two-dimensional field datasets.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problem-1-zero-error/requirements.txt
```

## Notebooks

| Notebook | Analysis | Output folder |
| --- | --- | --- |
| `01_solution_method_analysis.ipynb` | DP vs Greedy quality, timing, allocation agreement | `results/methods/` |
| `02_minimum_sensor_trends.ipynb` | Sequential minimum budget trends | `results/nmin/` |
| `03_global_seq_comparison.ipynb` | Greedy, sequential optimum, global minimum | `results/lb/` |
| `final_figures.ipynb` | Figures and tables from saved full results | `results/final/` |

Notebooks 01–03 run independently from saved inputs. Choose `RUN_MODE = "test"`
or `"full"` in each configuration cell, then run all cells:

- `test`: 132 cases across both dimensions and all families and K values,
  using M=3 or 6 and replicate 1.
- `full`: all 39,600 cases.

Each analysis saves separate `test/` and `full/` folders with `tables/` and
`figures/`. Notebook 01 also saves per-dataset details in `1d/tables/` and
`2d/tables/`. Run all three in full mode before running `final_figures.ipynb`.

In 02 and 03, `REUSE_SAVED_RESULTS = True` reuses matching DP results from 01
or earlier 02/03 runs. Metadata, freshness or input hashes, and allocation
feasibility are checked. Missing or stale results are computed normally.
Let 01 finish first for maximum reuse. Set the flag to `False` for fresh DP
solves. Notebook 03 computes Greedy and the global integer program for each case.

## Inputs

Each dataset contains Gaussian, sinusoidal, and quadratic fields with K=2–12,
M=1–12, and 50 realizations per family and (K,M): 19,800 cases per dimension.
Generate inputs without running solvers:

```bash
python problem-1-zero-error/scripts/generate_benchmarks.py all
```

Run the method comparison from the command line:

```bash
python problem-1-zero-error/scripts/evaluate_benchmarks.py --mode test --require-existing
# Use --mode full for all cases.
```

An optional input-only pilot uses K={10,12,14,16,18,20}, M={5,10,15,20}, and
three realizations per family and cell (216 cases per dimension):

```bash
python problem-1-zero-error/scripts/generate_benchmarks.py all --suite scale
```

Pilot inputs go to `inputs/scale_1d/` and `inputs/scale_2d/`. The main notebooks
use `benchmark_1d` and `benchmark_2d`; the standard evaluator has a K≤12 guard.

## Budget definitions

$N_{\mathrm{greedy}}^{\mathrm{seq}}$ is the sequential Greedy budget and
$N_{\min}^{\mathrm{seq}}$ is the sequential DP optimum. The global minimum is

$$N_{\min}^{\mathrm{global}}=\min_{\boldsymbol n\in\mathbb Z_{\geq0}^M}\sum_m n_m,
\qquad \sum_m n_m|r_{ijm}|\geq\pi\quad\text{for every }i<j.$$

Here $r_{ijm}$ is the principal phase gap. This minimum refers to the integer
program; its constraints alone do not certify a shared global input state.
Every case is checked for
$N_{\min}^{\mathrm{global}}\leq N_{\min}^{\mathrm{seq}}\leq N_{\mathrm{greedy}}^{\mathrm{seq}}$.

`src/lower_bound.py` uses SciPy/HiGHS with zero relative MIP gap and $10^{-9}$
feasibility tolerances. Integer allocations are independently checked to
$10^{-8}$ after normalization by $\pi$. Computation uses floating-point inputs.
Existing code and CSV identifiers such as `N_LB` remain for compatibility.

## Figures and tables

Notebook 01 reports Greedy sensor excess, worst-case allowance used, quality
heatmaps, and allocation agreement. “Same allocation” compares the full sensor
vectors; “same total budget” compares their sums. Different vectors can both
be optimal. For K=2, zero theoretical excess allowance is reported as 0% used.

Notebook 02 saves `case_budgets.csv` and `nmin_by_dimension_k.csv`.
Notebook 03 saves:

- `tables/case_budgets.csv`: all three budgets, allocations, timings, and diagnostics.
- `tables/nlb_vs_nmin_by_dimension_k.csv`: mean curves by dimension and K.
- `tables/nlb_vs_nmin_by_k_m.csv`: summaries at fixed K and M.
- `figures/nlb_vs_nmin_by_k.{pdf,png}`: all three curves for each dimension.

These existing output names remain compatible with saved-result reuse.
Full budget curves pool 1,800 cases per dimension and K; test curves pool 6.

Final figures use blue, red, and green, with yellow-to-red heatmaps. The bar,
guarantee, and heatmap figures and their tables use K=3–12 and M=2–12.
Other figures retain the full saved ranges. Dataset panels use (a)/(b) labels.

## Checks

```bash
python problem-1-zero-error/src/test_lower_bound.py -q
```

These checks are separate from notebook test mode. Main implementations and
plot helpers live in `src/`; input and evaluation entry points live in `scripts/`.
