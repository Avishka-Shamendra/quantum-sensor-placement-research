# Minimum sensors for a target sequential error

Open `01_extended_analysis.ipynb` and choose `RUN_MODE = "test"` or `"full"`. The notebook is supplied unexecuted. It reads the existing Problem 2 Extended datasets in `../inputs/benchmark_1d` and `../inputs/benchmark_2d`. No instances are generated or modified.

The default targets are 0.1, 0.2, 0.3, 0.4 and 0.5. Test mode reuses Problem 2's existing deterministic Extended sample (180 instances per dimension); full mode uses all 2,160 instances per dimension. Every instance is reused across targets.

## Search

The saved `minimum_sensors_for_zero_error` is the sequential perfect-discrimination upper bound. The original Problem 1 dynamic program reconstructs its allocation and covering location for every field pair, and its minimum must match the saved value. Problem 1 does not store a decision tree: a deterministic lexicographic elimination tree is built from its covering locations. The original Problem 2 full-tree evaluator, including off-pair outcomes, verifies that this fallback has zero error up to numerical precision. The reported fallback error is the evaluated value, not an assigned zero.

Each target has an independent bisection starting at low=0 and high=N_min_seq. The original VND + Beam Search is called at floor((low+high)/2). An actual error at most epsilon saves the allocation/tree and sets high=N; otherwise low=N. Search stops when high-low<=1. Zero-budget guessing is also checked explicitly. Equal priors and K>=3 make zero sensors infeasible for all default targets.

The original heuristic defaults are preserved: near-uniform initialization, transfer sizes (1,2,3), best-improvement descent, 30 neighborhood iterations, zero improvement threshold, and beam width 3. No existing algorithm or objective is altered. Actual error is recomputed independently for each candidate and for every returned solution, including cached solutions. A solution is reported only when its evaluated error is <= epsilon, without a relaxed feasibility tolerance.

These are heuristic feasible sensor counts, not certified minima. The heuristic's error need not be monotone in budget. A failed budget test can therefore skip a smaller feasible solution. The result is the smallest verified feasible budget visited by that target's bisection, including its perfect fallback.

## Execution and persistence

`RUN_ANALYSIS=True` performs missing work; `RESUME=True` resumes saved bisection states after each evaluated budget. Completed targets are reused. Test and full mode share matching per-instance caches. Original source files, extension search code, settings, CSV bytes and metadata fingerprint each cache. `RUN_ANALYSIS=False` requires complete matching caches and only reevaluates saved solutions and creates reports.

`TIME_LIMIT_SECONDS=None` preserves an unrestricted per-budget search. A positive limit may be selected. A timeout never marks a budget infeasible: the search stops with its last verified feasible result, and can retry that budget on resume. Such interrupted targets are saved explicitly and excluded from paired summary curves. Failures in reconstructing or validating the perfect fallback are recorded; no unverified fallback is reported. The original Problem 1 upper-bound reconstruction has no timeout and can be expensive for large instances.

Per-target runtime includes the measured perfect-allocation reconstruction and verification time plus all its budget-search/evaluation time. The shared perfect setup is charged once to each target to represent the cost of running that target independently. Resumed targets retain accumulated measured time; failed timed-out attempts are included. This is not notebook wall time. Setup and budget-search time are also reported separately.

## Results

The figure plots median verified feasible sensor count versus target error, with 1D and 2D panels and a 25th–75th percentile (interquartile) band. No instances are excluded because of extreme counts. This band describes the distribution of instance results, not confidence in an estimate. Numerical summaries retain the mean and its 95% field-family-stratified bootstrap confidence interval alongside the median and quartiles. The same complete set of instances is used across targets.

Summary tables report mean sensor count, confidence limits, mean runtime, mean achieved error and maximum achieved error for each target. Coverage reports exclusions. `solutions.json` saves full allocations, decision trees and search traces; `verified_solutions.csv` contains scalar results. Figures export as vector PDF and 600-dpi PNG with Times New Roman 12 pt text. Results stay inside this extension.

## Greedy comparison

The notebook can run the same independent binary budget search using the unchanged Problem 2 Greedy allocation and greedy decision-tree builder. It uses the same perfect fallback and full actual-error verification. Set `RUN_ANALYSIS=True` to compute missing Greedy results; the existing VND search module and its matching caches are preserved. No Greedy search is performed in saved-results mode. After a combined run, set `SAVED_RESULTS` to its printed output directory for later plotting-only sessions.

The figure uses green for VND + Beam Search and red dashed for Greedy, with each method's median and interquartile band. When both methods are available, comparisons use the same instances completed by both methods at every target. Greedy is not coupled to VND or its beam-search tree. Its sensor-by-sensor allocation can be slow at large midpoint budgets; the optional existing per-budget time limit applies to both methods. A timeout remains incomplete rather than being treated as infeasibility.
