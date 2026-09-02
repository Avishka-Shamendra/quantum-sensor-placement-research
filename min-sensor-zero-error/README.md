# Minimum-Sensor Zero-Error Discrimination

This directory contains the reproducible pipeline for comparing the paper's
**Exact Dynamic Programming** method with its **Greedy Gain-Based
Approximation** on one- and two-dimensional spatial fields.

## Recommended entry point

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r min-sensor-zero-error/requirements.txt
jupyter lab min-sensor-zero-error/01_zero_error_algorithm_comparison.ipynb
```

Run all notebook cells. Existing benchmark inputs are reused by default, both
algorithms are run on every case, and the combined tables and figures are written below
`min-sensor-zero-error/results/algorithm_comparison/`.

The same workflow can be run without Jupyter:

```bash
python min-sensor-zero-error/scripts/evaluate_benchmarks.py
```

## Benchmark design

The evaluation uses two separate datasets:

| Dataset | Spatial region | Design | Cases |
|---|---|---:|---:|
| `benchmark_1d` | `[0, 1]` | 3 families × 8 `K` × 8 `M` × 30 realizations | 5,760 |
| `benchmark_2d` | `[0, 1]^2` | 3 families × 8 `K` × 8 `M` × 30 realizations | 5,760 |

Both datasets use:

- Gaussian, sinusoidal, and quadratic field families;
- `K = 3, 4, ..., 10` candidate fields;
- `M = 3, 4, ..., 10` candidate sensing locations;
- 30 independent realizations in every family × `K` × `M` cell; and
- phase scale `c = 1`.

Candidate locations are sampled independently and uniformly from the relevant
region. A common set of `M` locations is used for all `K` fields in a case.
Every new realization receives new field parameters and new locations.

Generate the inputs without running the algorithms:

```bash
python min-sensor-zero-error/scripts/generate_benchmarks.py all
```

Use `--overwrite` only when intentionally regenerating an existing benchmark.
The raw CSV files are deterministic intermediates and are ignored by Git; the
fixed designs, parameter distributions, and seeds are committed in
`scripts/generate_benchmarks.py` and `src/field_models.py`.

## Field families

For 1D, `x_m = alpha_m`. For 2D,
`x_m = (alpha_m, beta_m)`. Coordinate and centre values lie in `[0, 1]`.

| Family | Shared parameter ranges |
|---|---|
| Gaussian | amplitude `[0.5, 1.5]`, width `[0.1, 0.3]`, centre in the sensing region |
| Sinusoidal | amplitude `[0.5, 1.5]`, frequency uniformly chosen from `{1, 2, 3}`, phase offset `[-pi, pi]`; 2D orientation `[0, pi)` |
| Quadratic | background `[-0.5, 0.5]`, curvature `[-1, 1]`, centre in the sensing region |

The 2D Gaussian is isotropic, the 2D sinusoid is a plane wave, and the 2D
quadratic divides squared distance by two so its scale remains comparable to
the 1D version. Exact formulas and realized parameters are stored in each
dataset manifest.

## Phase-gap convention

For fields `f_i` and `f_j` at location `x_m`, the input is

```text
r_ijm = Arg(exp(i c (f_i(x_m) - f_j(x_m)))) in (-pi, pi].
```

Only pairs with `i < j` are stored. Reverse phases are reconstructed as

```text
r_jim = principal_phase(-r_ijm).
```

The required sensor threshold is

```text
N_ijm = ceil(pi / abs(r_ijm)).
```

A zero phase gap is unusable at that location. Generation verifies that every
field pair is distinguishable at least somewhere in each case.

## Evaluation outputs

Both problem folders use the same main comparison layout:

```text
results/algorithm_comparison/
├── benchmark_1d_results_full/
│   └── tables/
├── benchmark_2d_results_full/
│   └── tables/
└── results_summary_benchmark_1d_benchmark_2d_full/
    ├── tables/
    └── figures/
```

The `_full` suffix shows that this pipeline evaluates every generated case.

The primary solution-quality measure is

```text
rho = N_greedy / N_optimal.
```

The analysis also reports exact-optimum recovery, percentage additional
sensors, the median and 95th percentile ratio, the fraction of the theorem's
permitted excess used, and runtime. Results are kept separate by spatial
dimension as well as summarized across both benchmarks.

Three publication figures are generated, each with side-by-side 1D and 2D
panels:

- mean sensor excess over `K` and `M`;
- mean sensor excess by `K` and field family, with 95% bootstrap intervals;
  and
- family-specific maximum observed use of the theoretical worst-case excess
  allowance.

The files are named `benchmark_quality_heatmaps.*`,
`benchmark_family_excess.*`, and `benchmark_guarantee_comparison.*`. Each is
saved as a vector PDF and a 300-dpi PNG. Runtime and sample-convergence values
remain available in the CSV summaries but are not exported as additional
figures.

For final runtime measurements, use repeated solves on an otherwise idle
machine:

```bash
python min-sensor-zero-error/scripts/evaluate_benchmarks.py --repeats 5
```

## Code layout

```text
min-sensor-zero-error/
├── 01_zero_error_algorithm_comparison.ipynb
├── src/
│   ├── optimal.py             # Exact Dynamic Programming
│   ├── greedy.py              # Greedy Gain-Based Approximation
│   ├── phase_data.py          # phase validation and threshold conversion
│   ├── field_models.py        # agreed 1D and 2D field distributions
│   ├── eval.py                # evaluates both algorithms on one dataset
│   └── paper_analysis.py      # statistics, tables, and publication figures
├── scripts/
│   ├── generate_inputs.py     # reusable dataset writer
│   ├── generate_benchmarks.py # fixed benchmark definitions
│   ├── evaluate_dataset.py    # one-dataset command-line evaluation
│   └── evaluate_benchmarks.py # complete command-line workflow
├── docs/
│   └── evaluation_strategy.md
├── inputs/                    # generated benchmark inputs
└── results/                   # comparison tables and figures
```

Reusable implementation remains in Python modules. The notebook imports those
modules and acts as the documented experiment runner.

## Small custom datasets

For a quick custom test rather than the full benchmarks:

```bash
python min-sensor-zero-error/scripts/generate_inputs.py custom_2d \
  --dimension 2 --cases 20 --k-min 3 --k-max 6 \
  --m-min 3 --m-max 6 --c 1 --seed 20260728
python min-sensor-zero-error/scripts/evaluate_dataset.py custom_2d
```

Exact evaluation is limited to `K <= 10` because the reference dynamic
program is exponential in the number of field pairs.
