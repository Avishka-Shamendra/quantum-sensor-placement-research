# Sequential vs. Joint Minimum-Error Discrimination

Open `01_sequential_vs_joint_analysis.ipynb` from this directory or the repository
root. Set `RUN_MODE = "full"` for the requested study, then run all cells manually.
The default test mode uses K, M in {2, 3} and two replicates per cell; it uses the
same algorithms and budget rules. Dependencies are listed in `requirements.txt`.

Full mode draws 7,290 independent instances: K and M from 2 through 10, three
families, two dimensions, and 15 replicates per cell. Inputs use the original
field generators, parameter ranges, signed principal phases, c = 1, and current
ordinary-random dataset validity policy (minimum gap cutoff zero). CSV files,
seeds, parameters, sensing locations, source hashes, and exclusions are saved.
No draw is selected or replaced based on either method's performance. Exactly
indistinguishable pairs and instances without a positive budget are excluded;
solver failures stop execution instead of silently selecting different inputs.

The original joint ILP gives a necessary lower bound for perfect discrimination,
not a guarantee for the chosen probe. Budgets use
`N_max = min(N_LB_joint - 1, 50)`. Fractions 0.1–0.9 are rounded to the nearest
integer, with halves upward; zero counts are omitted. Each distinct count is
run once per instance and retains all its target-fraction aliases. A positive
rounded count can equal N_max, which remains strictly below the lower bound.

The sequential adapter calls the unchanged VND + Beam Search implementation
(beam width 3, original 30-iteration default, configurable) and explicitly
reevaluates its complete decision tree with off-pair probabilities. Its seeded
near-uniform initialization is reproducible per instance. The joint adapter
calls the unchanged VND allocation search, then evaluates the local equal-Dicke-
sector superposition with PGM. Both use equal hypothesis priors and identical
CSV inputs and total sensor counts. Neither plotted quantity is an allocation
score. These are the original methods' final errors, not certificates of global
minimum error.

Figure 1 averages matched instances separately at each target fraction. Repeated
integer counts are mapped back to their target fractions without rerunning them.
Zero-rounded targets contribute no observation; missing points remain gaps.
Pointwise 95% percentile intervals use 2,000 seeded instance bootstrap resamples
within field-family/K/M strata, with the same resampling indices for both methods.
Singleton strata contribute no estimated variation; test-mode intervals are
previews. These are not simultaneous confidence bands.

Table 1 first averages the unique positive budgets within each matched instance,
then averages those instance means within dimension/family. This gives every
instance equal weight and avoids counting duplicate rounded budgets multiple
times. Both outputs use precisely the same paired result records. A failure
stops the pipeline before publishing incomplete results; completed methods remain
cached for the next run.

The figure follows Problem 3's publication notebook: Times New Roman 12 pt,
10.8 × 4.6 inches, a shared lower legend, embedded PDF fonts, and 600-dpi PNG.
Sequential retains Problem 2's green triangle heuristic style; Joint retains
Problem 3's purple diamond local-Dicke style. Install Times New Roman locally to
avoid Matplotlib's font fallback.

`data/<mode>/<fingerprint>/` contains reproducible inputs and the manifest.
`results/<mode>/<fingerprint>/` contains atomic per-method checkpoints, matched
results, provenance, Figure 1 (PDF/PNG and numerical estimates), and Table 1
(CSV/Markdown). Interrupted runs resume automatically. Changed settings or source
hashes select separate caches. Do not edit source files in an active kernel;
restart the kernel after source changes. Original projects are imported in place
and never modified. No experiment or notebook was executed during implementation.
