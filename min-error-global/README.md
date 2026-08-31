# Minimum-error global discrimination

This directory evaluates fixed-budget sensor allocations for the three global
input-state models in Section V:

- Product sensors
- Location-wise GHZ states
- One global GHZ state

Every algorithm minimizes or evaluates the actual equal-prior Pretty Good
Measurement (PGM) error obtained from the output-state Gram matrix.

## Inputs

The global benchmarks are generated independently from the paper's Gaussian,
sinusoidal, and quadratic field families. They store

```text
delta_ijm = pi * (f_i(x_m) - f_j(x_m))
```

without wrapping, clipping, rejection, or resampling. Since the field-value
difference interval is `[-3, 3]`, the theoretical phase-difference interval is
`[-3*pi, 3*pi]`. Zero-valued differences are retained.

Generate all compact or extended benchmarks with:

```bash
.venv/bin/python min-error-global/generate_compact_benchmarks.py all --overwrite
.venv/bin/python min-error-global/generate_benchmarks.py all --overwrite
```

## Algorithms

Only algorithm implementations live in `algorithms/`:

- `optimal.py`: exhaustive minimization of actual PGM error over every weak
  composition of sensor budget `N` into `M` locations.
- `greedy.py`: adds one sensor at a time at the location giving the smallest
  actual PGM error.
- `near_uniform.py`: gives every location `N // M` sensors and distributes the
  `N % M` remainder sensors among distinct randomly selected locations.
- `random_allocation.py`: samples uniformly from all weak compositions of `N`
  into `M` locations.

Randomized methods require an explicit seed. The comparison runner derives
stable seeds from the dataset, case, budget, and method and reuses each sampled
allocation across the three state models.

## Comparison notebooks

Run [comapre.ipynb](comapre.ipynb) to compare all four methods for sensor
budgets `N=3,...,12`. The `RUN_ALL_CASES` boolean selects either all compact
instances or a reproducible test sample of 100 instances per dataset. This is
the exact-optimality experiment.

The notebook produces six images, all using 12-point Times New Roman labels:

1. Mean actual PGM errors in a six-panel state-by-dimension figure.
2. Mean actual PGM-error gaps from Optimal in another six-panel figure.
3. PGM-error gaps grouped by field family in a six-panel figure.
4. Greedy input-state comparison with separate 1D and 2D panels.
5. Optimal input-state comparison with separate 1D and 2D panels.
6. Paired runtime-speedup box plots for all four methods with separate 1D and
   2D panels. Speedup is `Optimal runtime / method runtime` on a logarithmic
   axis.

JSON, CSV, PNG, and PDF outputs are written beneath:

```text
min-error-global/results/actual_pgm_four_methods_c_pi/
```

The end-of-run diagnostics report mean, 90th-percentile, and maximum gaps;
relative gaps; exact-optimality fractions; and observed `delta_ijm` and
`N * delta_ijm` ranges.

Run [compare_extended.ipynb](compare_extended.ipynb) for the larger
`benchmark_1d` and `benchmark_2d` designs. It compares Greedy, Near-uniform,
and Random without exhaustive Optimal and generates:

1. A six-panel mean actual PGM-error figure.
2. A six-panel actual PGM-error-by-field-family figure.
3. A two-panel raw-runtime box plot.

Extended outputs are written beneath:

```text
min-error-global/results/actual_pgm_scalable_methods_c_pi_extended/
```

For the main paper, use the compact gap-to-Optimal figure, the compact
field-family gap figure, and the extended mean-error figure. Treat runtime as
optional and keep the remaining plots in supplementary material.
