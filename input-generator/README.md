# Input Dataset Generator

Generate one dataset folder below `inputs/` from a selected spatial field
family. The available families are `gaussian`, `sine`, `fourier`, and
`polynomial`.

```bash
python3 input-generator/generate_dataset.py \
  --dataset gaussian-study \
  --family gaussian \
  --k-min 3 --k-max 10 \
  --m-min 3 --m-max 10 \
  --cases 20
```

The generator cycles through a shuffled K/M grid, so the requested ranges are
covered as evenly as the number of cases allows. It creates CSV files and a
`manifest.json` containing the seed, K, M, sensing locations, field family,
phase-scale constant, analytic formula, and each candidate field's exact
parameters (for example, `A_i`, `mu_i`, and `sigma_i` for Gaussian fields).
The manifest is the record of the functions tested; the CSV remains the
numerical Algorithm 1/2 input only.

`c` is fixed at `1.0` by default. Use one common value for all datasets that
will be compared:

```bash
python3 input-generator/generate_dataset.py \
  --dataset sine-study --family sine \
  --k-min 3 --k-max 8 --m-min 3 --m-max 8 --cases 12 --c 1.0
```

Use `--overwrite` only when intentionally replacing an existing generated
dataset with the same name.

## Four matching example datasets

These commands generate one 50-case dataset for each field family, each
covering `3 <= K <= 10` and `3 <= M <= 10`. They hold the phase scale and
base seed fixed so the studies are comparable. The generator derives a
different deterministic per-case seed within each dataset.

```bash
python3 input-generator/generate_dataset.py \
  --dataset gaussian-k3-10-m3-10 --family gaussian \
  --k-min 3 --k-max 10 --m-min 3 --m-max 10 --cases 50 \
  --c 1.0 --seed 20260728

python3 input-generator/generate_dataset.py \
  --dataset sine-k3-10-m3-10 --family sine \
  --k-min 3 --k-max 10 --m-min 3 --m-max 10 --cases 50 \
  --c 1.0 --seed 20260728

python3 input-generator/generate_dataset.py \
  --dataset fourier-k3-10-m3-10 --family fourier \
  --k-min 3 --k-max 10 --m-min 3 --m-max 10 --cases 50 \
  --c 1.0 --seed 20260728

python3 input-generator/generate_dataset.py \
  --dataset polynomial-k3-10-m3-10 --family polynomial \
  --k-min 3 --k-max 10 --m-min 3 --m-max 10 --cases 50 \
  --c 1.0 --seed 20260728
```

Change `--cases` if you need more or fewer instances. Use a different
`--seed` only when you intentionally want a new realization of the study.
