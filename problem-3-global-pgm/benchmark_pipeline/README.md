# Problem 3 benchmark pipeline

The active Compact dataset is `inputs/compact`; Extended remains `inputs/extended`. Both studies include 1D and 2D. Generation uses ordinary independent random draws from the existing Gaussian, sinusoidal, and quadratic parameter distributions and uniform sensing locations. Each candidate has a recorded deterministic RNG seed. There is no positive minimum-gap threshold. Compact generation rejects only exactly indistinguishable field pairs; Extended copies every existing Problem 2 Extended instance without filtering. No allocation search or landscape test participates in selection.

Compact uses K,M in {3,4,5}, with 50 instances per family/configuration/dimension (2,700 total). Extended uses K in {3,...,10} and M in {2,...,10}, with 10 instances per cell (4,320 total), identical to Problem 2. Budgets are the distinct ceilings of {0.25,0.5,0.75,1} times min(N_LB, cap), with caps 16 and 50 respectively. Actual budgets and N/N_LB ratios are recorded. The joint lower bound is certified by the existing integer program; a numerical certification failure retries without solver presolve and with equivalent positive row scaling, preserving the original feasibility and optimality checks. Such failures never reject a candidate on difficulty grounds.

CSV files retain `i,j,m,r_ijm` with canonical wrapped signed phase gaps. Manifests additionally store the actual field values, parameters and locations, so heuristic phase differences are computed directly from the original fields with the encoding factor included. Field instances are reused across all budgets, algorithms and states. Only the final Compact and Extended datasets are retained in `inputs`.

From the Problem 3 directory:

```bash
../.venv/bin/python benchmark_pipeline/generate.py --output /tmp/problem3-regenerated --studies compact --compact-replicates 50 --workers 4
../.venv/bin/python benchmark_pipeline/validate.py inputs/compact
```

Extended CSVs are copied byte-for-byte from `problem-2-min-error-seq/inputs/benchmark_1d` and `benchmark_2d`. Both problems use the shared Problem 1 field generators; actual field values are reconstructed from Problem 2 recorded seeds and verified against the CSVs. Only the joint bound and budget metadata are recomputed. Source manifests and CSVs are fingerprinted, and every copied CSV records its source and SHA-256. Generation resumes completed configuration checkpoints only when settings and source fingerprints match. Validation regenerates every input from its seed, verifies all rejection decisions, checks exact CSV values and budget schedules, and independently resolves the joint lower bounds.

## Reporting

`publication.py` creates exactly two figures per notebook. Compact compares the five states using Exhaustive and shows heuristic gaps relative to Exhaustive. Extended compares states using VND and shows grouped error-difference bars relative to VND, which is not a certified optimum. Figures use Times New Roman 12 pt, 95% field-instance bootstrap intervals, vector PDF and 600-dpi PNG. Numerical summaries report overall and field-family results.

Allocation time is recorded once per heuristic/instance/budget. Each state receives the same allocation, one final PGM evaluation, and one fifth of the allocation runtime in its amortized total. Runtime tables separately report allocation and final evaluation times. Exhaustive reports its complete objective-driven enumeration time. Reference baselines are enabled by default and include Near-Uniform and the average over 20 reproducibly seeded uniform weak compositions to the existing gap figure.
