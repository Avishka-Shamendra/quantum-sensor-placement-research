# Single-location state and measurement optimization

## Execution

Open `01_single_location_analysis.ipynb` from this directory or the repository root and run the cells in order. The setup cell installs missing dependencies into the active kernel. Alternatively, install them with `python -m pip install -r requirements.txt`.

The notebook defaults to `RUN_ANALYSIS=False` and reports from saved measurements and weights. Set `RUN_ANALYSIS=True` only to perform missing evaluations. Analysis runs display progress and save completed evaluations for resumption. Use one writer per checkpoint directory.

## Experiment

M = 1; K = {3, 4, 5}; N = {2, 4, 6, 8, 10}. Each field family—Gaussian, sinusoidal and quadratic—has 30 instances per K: 270 field instances and 1,350 instance/budget cases. All budgets share the same field instances. Archived K = 2 and odd-budget cases are retained but excluded from analysis.

Five strategies are compared: optimized Dicke-sector weights with optimal POVM or PGM, and product, GHZ or equal-sector states with PGM. Optimized weights minimize the numerical SDP objective using multi-start SLSQP; they are not certified global optima. The optimized-state PGM curve uses the same weights, without a separate PGM optimization.

States have amplitudes sqrt(p_w) in the Dicke basis and acquire phases exp(i w phi_j). Field priors are uniform. Product weights are binomial; GHZ uses equal endpoint weights; equal-sector weights are 1/(N+1). Permutation symmetry and reflection symmetry p_w = p_(N-w) are distinct. Reflection symmetry is not imposed during optimization.

## Data and validation

`imported/` contains the original datasets, configurations and saved results. `manifest.json` records archive paths and checksums. All 1,350 requested optimized weight vectors are available. The original Gaussian `pgm.csv` labels optimized-state SDP errors as `opt_sdp`; these are not optimized-state PGM errors.

The first run validates archived weights and recomputes fixed-state SDPs. Subsequent runs reuse validated checkpoints. Checks cover normalization, probability bounds, SDP dual feasibility and PGM ≥ optimal-POVM error, with tolerance 1e-5. Tables report archived-error discrepancies and optimizer convergence. These checks provide numerical validation, not exact certificates. Validation failures stop execution.

Missing or invalid optimized weights use the original settings: eight SLSQP starts, 120 iterations, ftol = 1e-8, SDP tolerance = 1e-7, and seed sequence [20260930, K, instance, N]. Ancillary annealing checks are retained in the archived configuration but are not run. Cache signatures include source code, inputs and environment versions. Each state evaluation and optimized weight vector is saved separately.

## Results

Each figure has three panels: Gaussian, sinusoidal and quadratic. Figures 1–2 average equally over K = 3, 4, 5, with 30 instances per K. Figure 3 retains individual points from all K values. Figure-summary tables provide stratified Welch 95% intervals for the equal-weight means; the separate K tables are retained. M = 1 is fixed; no other location counts are available.

- Figure 1: mean optimal-POVM error versus N for four input states.
- Figure 2: mean paired PGM-minus-optimal-POVM error versus N for the same states.
- Figure 3: optimized-sector probabilities p_w versus p_(N-w), colored by N. Each point represents a pair of distinct reflected sectors from one instance; the diagonal indicates reflection symmetry. Central sectors are excluded.

Figures show no confidence bands. The optimal-POVM and measurement-penalty tables report means and 95% Student-t intervals across 30 instances for every family, K, M = 1, N and state. The symmetry table summarizes the per-instance maximum absolute reflected-weight difference, including the fraction within 1e-5. Sector-weight means and confidence intervals, per-instance symmetry values, runtimes and numerical validation are exported separately. Symmetry is measured on the saved weights without symmetrization or averaging across instances before calculating the differences.

Outputs are saved in `results/` as PDF, 600-dpi PNG and CSV, using Times New Roman at 12 pt. The numerical implementations are in `single_location/`; lightweight checks are in `tests/`.
