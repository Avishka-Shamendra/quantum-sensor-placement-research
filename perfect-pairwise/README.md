# Perfect Pairwise Discrimination

Run all commands from the repository root.

## Create an input dataset

The configurable generator is in [`input-generator`](../input-generator/).
Choose exactly one spatial-field family per dataset: `gaussian`, `sine`,
`fourier`, or `polynomial`. For example, this creates 20 Gaussian cases
with `3 <= K <= 10` and `3 <= M <= 10`:

```bash
python3 input-generator/generate_dataset.py \
  --dataset gaussian-small \
  --family gaussian \
  --k-min 3 --k-max 10 \
  --m-min 3 --m-max 10 \
  --cases 20
```

The cases are deterministically spread across the requested K/M grid. The
generator saves them as `inputs/gaussian-small/*.csv`, together with a
`manifest.json` that records the analytic formula, exact per-field
parameters, family, seed, locations, and phase-scale constant.

Each CSV contains `i,j,m,r_ijm` rows for `i < j` only. The generator obtains
them from the Section III relation

```text
r_ijm = principal_value(-c (f_i(x_m) - f_j(x_m))).
```

## The sensing constant c

You do not need to calculate `c` for every CSV: the generator applies it
internally before writing `r_ijm`. Its default is `c = 1.0`, and that value
is recorded in the manifest. Keep the same `c` across datasets you compare.
If your physical model supplies a calibrated value, pass it explicitly:

```bash
python3 input-generator/generate_dataset.py \
  --dataset sine-small --family sine \
  --k-min 3 --k-max 10 --m-min 3 --m-max 10 --cases 20 \
  --c 1.0
```

Once a CSV exists, `c` has already been absorbed into `r_ijm`; the evaluator
therefore needs only the CSV.

Use `--overwrite` only when deliberately replacing a dataset with the same
name.

## Evaluate

Run both algorithms for a dataset whose cases all have `K <= 10`:

```bash
python3 perfect-pairwise/eval.py gaussian-small --mode both --repeats 100
```

For a larger-K dataset, run Algorithm 2 only:

```bash
python3 perfect-pairwise/eval.py fourier-large --mode greedy --repeats 100
```

`--mode auto` uses both algorithms for `K <= 10` and greedy only otherwise.
It rejects an exact (`both` or `optimal`) request if any case has `K > 10`.

```bash
python3 perfect-pairwise/eval.py all --mode auto
```

The evaluator writes `perfect-pairwise/output/<dataset>/` containing:

- `nmin_comparison.csv` and a printed terminal table with Nmin and timing.
- `algorithm_quality.svg` when the exact solution is available.
- `greedy_scalability.svg` for greedy-only data.
- `family_summary.csv` plus complete per-case JSON and CSV records.

When you evaluate `all` datasets with multiple field families, the paper
figure to use is written once at
`perfect-pairwise/output/cross-family-summary/cross_family_quality.svg`.
It has four matched approximation-ratio panels (Gaussian, sine, Fourier,
and polynomial). Each panel includes the dotted theorem bound on a labelled
right-hand axis, rather than repeating a separate bound-only plot. The same
folder contains a combined `case_results.csv` and four-row
`family_summary.csv`.
