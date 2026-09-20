# Minimum-Sensor Zero-Error Discrimination

This folder contains the zero-error sensor-allocation pipeline. It generates
one- and two-dimensional benchmark inputs, evaluates exact dynamic programming
and greedy approximation methods, and summarizes the resulting sensor budgets.

## Structure

```text
problem-1-zero-error/
├── 01_zero_error_algorithm_comparison.ipynb
├── src/
│   ├── optimal.py             # exact dynamic program
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

Run the full benchmark workflow:

```bash
python problem-1-zero-error/scripts/evaluate_benchmarks.py
```

Or run the notebook:

```bash
jupyter lab problem-1-zero-error/01_zero_error_algorithm_comparison.ipynb
```

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

```text
rho = N_greedy / N_optimal
```

where `N_optimal` is the minimum sensor budget returned by the exact dynamic
program and `N_greedy` is the budget returned by the greedy method.

## Outputs

Generated files are written under `problem-1-zero-error/results/`:

```text
results/algorithm_comparison/
├── benchmark_1d_results_full/
├── benchmark_2d_results_full/
└── results_summary_benchmark_1d_benchmark_2d_full/
```

Tables are saved as CSV files and figures are saved as PDF and PNG files.
