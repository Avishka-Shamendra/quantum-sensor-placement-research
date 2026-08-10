# Evaluation Strategy for the Paper

## Research question

The evaluation asks how closely the greedy allocation follows the exact
minimum sensor allocation on the stated synthetic spatial-field distribution.
Every comparison is paired: both algorithms receive exactly the same phase-gap
instance.

## Pre-specified benchmark datasets

The spatial dimensions are kept as two distinct datasets so their results can
be reported separately.

| Dataset | Dimension | Field families | `K` | `M` | `c` | Realizations per cell | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `benchmark_1d` | 1 | 3 | 3--10 | 3--10 | 1 | 30 | 5,760 |
| `benchmark_2d` | 2 | 3 | 3--10 | 3--10 | 1 | 30 | 5,760 |

The three families are Gaussian, sinusoidal, and quadratic. They represent a
localized field, a periodic field, and a smooth nonperiodic trend. They are a
transparent synthetic test distribution, not a standard measured benchmark.
The complete formulas and sampling ranges are defined in `field_models.py`.

For each realization, new field parameters and new sensing locations are
sampled. The 1D locations are independent uniform samples from `[0, 1]`; the
2D locations are independent uniform samples from `[0, 1]^2`. All `K` fields
in a case are evaluated on the same set of `M` locations.

Each dataset contains 3 families × 8 values of `K` × 8 values of `M` = 192
factorial cells. Thirty realizations per cell give 5,760 cases per dimension
and 11,520 cases overall.

## Outcomes

The primary outcome is the greedy-to-optimal sensor ratio

```text
rho = N_greedy / N_optimal.
```

Report the mean ratio with a factorial-cell-stratified bootstrap interval,
median, 95th percentile, maximum, and exact-optimum recovery rate. Also report
the percentage additional sensors used by greedy and solver runtimes.

For `K` fields, the theorem's worst-case factor is

```text
B(K) = 1 + ln(K(K - 1) / 2).
```

The useful empirical comparison is

```text
(rho - 1) / (B(K) - 1),
```

which measures the fraction of the additional sensors allowed by the bound
that greedy actually used.

## Reporting structure

The main results use three figures, each with side-by-side 1D and 2D panels:

1. a `K` by `M` heatmap of mean percentage sensor excess;
2. grouped bars of mean percentage sensor excess by `K` and field family,
   with 95% bootstrap intervals stratified by `M`; and
3. the family-specific maximum observed fraction of the logarithmic
   worst-case excess allowance used by greedy.

Use identical axes and heatmap scales for the 1D and 2D panels. Report the
exact-match rates and 30-realization convergence diagnostic numerically rather
than as additional figures. Report a combined summary only after the
dimension-specific results.

The intended conclusion is limited to the generated distribution: greedy can
be described as empirically closer to optimal than the worst-case allowance
on these normalized synthetic fields. The experiment does not improve the
theorem and cannot by itself establish performance for every physical sensing
application.

## Reproducibility

The fixed benchmark definitions and seeds are in `generate_benchmarks.py`.
Every realized location and field parameter is recorded in the corresponding
manifest. Raw case files and solver outputs are deterministic, ignored
intermediates that can be recreated with:

```bash
python min-sensor-zero-error/generate_benchmarks.py all
python min-sensor-zero-error/evaluate_benchmarks.py --require-existing
```

For manuscript runtime measurements, repeat each solve on an idle machine and
report the processor, operating system, Python version, and repetition count.
