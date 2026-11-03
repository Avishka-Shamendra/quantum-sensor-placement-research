# Sequential vs. joint minimum-error discrimination

Matched comparison for paper §6. The notebook now defaults to **Problem 2 VND + Beam Search versus Problem 3 Greedy**, with the local equal-Dicke-sector superposition and PGM. Both methods use identical inputs, priors and sensor counts. The underlying search procedures are unchanged.

## Configuration

Open `01_sequential_vs_joint_analysis.ipynb` and run cells in order:

- `RUN_MODE='full'`: K,M=2–10, three families, two dimensions, 15 draws per cell (7,290 instances before exclusions).
- `RUN_MODE='compact'`: the previous test dataset, K,M=2,3 and two draws per cell. `test` remains an alias.
- `SEQUENTIAL_ALGORITHM='vnd_beam'` or `'greedy'`.
- `JOINT_ALGORITHM='greedy'` or `'vnd'`.
- `RUN_ANALYSIS=True` computes only missing checkpoints. `False` requires all selected checkpoints and raises on missing work without launching a solver.

The input cell shows cache coverage before evaluation. Existing datasets and their checksums are reused independently of search/report settings. A new seed selects a new dataset. Stored manifests freeze the generated inputs and budget schedules.

## Saved results

Reports overwrite `results/full/` or `results/compact/` directly; there are no hashed report-run directories. `provenance.json` identifies the algorithms used for the current exports. Plot labels identify both algorithms; mismatched row/config labels are rejected.

Checkpoints remain separate under `results/<mode>/cache/<side>/<algorithm>/<instance>/`. Their filenames fingerprint the relevant solver/evaluator sources, input checksum and search settings. Changing the joint algorithm does not invalidate sequential checkpoints; switching back reuses the prior joint results. Changing bootstrap settings only rebuilds reports. Runtime measurements are retained and exported in `matched_results.csv`.

The legacy VND comparison checkpoints were migrated after checking input identity, allocations and occupied tree locations. Original joint VND functions and evaluators matched exactly; sequential changes were diagnostics and occupied-location guards. `cache_migration.json` records reused/rejected counts. Superseded legacy artifacts have been removed. Historical flat exports still describe VND until a new selected comparison completes; they are not Greedy results.

Data remain in `data/<dataset-mode>/<fingerprint>/`; report-folder cleanup does not regenerate data. Do not delete the `cache` directories if you want to resume.

## Budgets and statistics

Budgets are rounded fractions 0.1–0.9 of `min(N_LB_joint - 1, 50)`, with halves rounded upward. Zero counts are dropped and duplicate counts evaluated once. Sequential error includes off-pair outcomes; joint error is final PGM error, not phase score.

Figure 1 averages matched errors at each target fraction with pointwise stratified bootstrap intervals. Table 1 averages unique budgets within each instance, then instances within dimension/family. Win rates count each unique instance/budget once, with tie tolerance 1e-9. Exports include PDF, PNG, CSV and Markdown.
