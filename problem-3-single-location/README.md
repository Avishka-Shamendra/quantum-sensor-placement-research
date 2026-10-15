# Single-location state and measurement optimization

## Execution

Open `01_single_location_analysis.ipynb` from this directory or the repository root and run the cells in order. The setup cell installs missing dependencies into the active kernel. Alternatively, install them with `python -m pip install -r requirements.txt`.

The status line reports the current family, K, instance, budget and state, followed by completed, cached and newly evaluated counts. Completed evaluations are saved automatically. Restarting the notebook resumes from validated checkpoints. Use one writer per checkpoint directory.

## Experiment

M = 1; K = {3, 4, 5}; N = {2, 4, 6, 8, 10}. Each field family—Gaussian, sinusoidal and quadratic—has 30 instances per K: 270 field instances and 1,350 instance/budget cases. All budgets share the same field instances. Archived K = 2 and odd-budget cases are retained but excluded from analysis.

Five strategies are compared: optimized Dicke-sector weights with optimal POVM or PGM, and product, GHZ or equal-sector states with PGM. Optimized weights minimize the numerical SDP objective using multi-start SLSQP; they are not certified global optima. The optimized-state PGM curve uses the same weights, without a separate PGM optimization.

States have amplitudes sqrt(p_w) in the Dicke basis and acquire phases exp(i w phi_j). Field priors are uniform. Product weights are binomial; GHZ uses equal endpoint weights; equal-sector weights are 1/(N+1). Permutation symmetry and reflection symmetry p_w = p_(N-w) are distinct. Reflection symmetry is not imposed during optimization.

## Data and validation

`imported/` contains the original datasets, configurations and saved results. `manifest.json` records archive paths and checksums. All 1,350 requested optimized weight vectors are available. The original Gaussian `pgm.csv` labels optimized-state SDP errors as `opt_sdp`; these are not optimized-state PGM errors.

The first run validates archived weights and recomputes fixed-state SDPs. Subsequent runs reuse validated checkpoints. Checks cover normalization, probability bounds, SDP dual feasibility and PGM ≥ optimal-POVM error, with tolerance 1e-5. Tables report archived-error discrepancies and optimizer convergence. These checks provide numerical validation, not exact certificates. Validation failures stop execution.

Missing or invalid optimized weights use the original settings: eight SLSQP starts, 120 iterations, ftol = 1e-8, SDP tolerance = 1e-7, and seed sequence [20260930, K, instance, N]. Ancillary annealing checks are retained in the archived configuration but are not run. Cache signatures include source code, inputs and environment versions. Each state evaluation and optimized weight vector is saved separately.

## Results

- Figure 1: mean discrimination error versus N for five strategies at K = 4, with one panel per field family.
- Figure 2: paired PGM-minus-optimal-POVM error versus N for four state choices at K = 4, with one panel per field family.

Figures show mean curves without shaded bands. Tables include pointwise 95% Student-t intervals across 30 independent instances. Tables cover every K, family and budget, including measurement degradation, runtime and validation summaries. Runtime summaries distinguish historical optimization time from measurement-evaluation time; cache-loading time is excluded.

Outputs are saved in `results/` as vector PDF, 600-dpi PNG and CSV. Figures request Times New Roman with a serif fallback. The numerical implementations are in `single_location/`; lightweight algebraic tests are in `tests/`.
