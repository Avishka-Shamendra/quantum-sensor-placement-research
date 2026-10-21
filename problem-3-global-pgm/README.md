# Problem 3: allocation with global PGM evaluation

`01_compact_analysis.ipynb` compares Exhaustive, Greedy and VND. `02_extended_analysis.ipynb` compares Greedy and VND without Exhaustive. Compact uses `inputs/compact`; Extended uses `inputs/extended`. Both support test/full mode and resumable matching results. Neither notebook is executed during preparation.

## Algorithms

For every field pair, d_ijm is the absolute actual encoded phase difference. Greedy starts at zero, selects the currently least-separated pair, and adds a sensor at that pair's strongest location. Tied pairs prefer the smaller maximum separation, then lexicographic pair order; tied locations prefer the smallest index.

VND independently starts near uniformly, assigning remainders to the first locations. It maximizes S(n) = mean_ij min(pi, sum_m n_m d_ijm), scanning best-improvement transfers of one, two, and three sensors. Improvements greater than 1e-12 restart the one-sensor neighborhood. Source/destination iteration order breaks ties deterministically. VND stops at convergence; the prior 30-iteration cap is superseded by this specification. A timeout is reported as incomplete.

Neither heuristic evaluates PGM during search. One allocation per instance and budget is applied to all five unchanged probe families. Their final actual global PGM errors are evaluated separately. Improving the phase score does not guarantee improving PGM error.

Exhaustive enumerates every feasible allocation separately for each probe state, retaining the first minimum in allocation enumeration order. It is optimal over allocations for that state and PGM measurement, not over all quantum measurements.

## Results

Exactly four figures are produced: Compact state curves using Exhaustive and heuristic gap bars; Extended state curves using VND and grouped error-difference bars relative to VND. Each figure has 1D/2D panels, 95% confidence intervals, Times New Roman 12 pt text, PDF and PNG exports. Overall and field-family tables include runtime, error, Compact optimality-match and VND win/tie/loss summaries. Random and Near-Uniform references are enabled by default. Both second figures are grouped bars by input state, with 95% confidence intervals; negative Extended differences indicate lower error than VND.

The dataset generator and validation commands are documented in [benchmark_pipeline/README.md](benchmark_pipeline/README.md). Results use fixed `results/{compact,extended}/{test,full}/` folders. Fresh runs overwrite that folder; `RESUME=True` reuses matching results only. Source, settings and dataset fingerprints remain internal checks against stale results. The final-figures notebook reads the two `full/` folders directly, without run IDs.

Extended uses the exact Problem 2 Extended CSV instances: K=3–10, M=2–10, 10 replicates per family/configuration/dimension, 4,320 total. Problem 3 retains joint-lower-bound budgets and normalization. Compact retains K,M=3–5 and 50 replicates per cell and 2,700 instances. The positive 0.01 gap filter is removed; exactly indistinguishable pairs remain invalid for a finite joint bound.

Greedy and VND take signed `r_ijm` values directly from the phase CSV. Greedy selects the pair minimizing `sum_m n_m*abs(r_ijm)` and adds a sensor at its location maximizing `abs(r_ijm)`. VND uses the mean capped separation `mean_{i<j} min(pi, sum_m n_m*abs(r_ijm))`. Manifest field values are not algorithm inputs. Result policy `csv_rijm_pair_greedy_score_vnd_v3_single_random` separates these results from earlier allocations based on unwrapped fields.

Allocation-comparison bars in the Compact and Extended analyses and final figures show 95% percentile bootstrap confidence intervals (2,000 resamples, seed 20261014). Differences are paired within instance/budget/state, then averaged over budgets within each instance. Bootstrap sampling is stratified by K, M, and field family. Final figure CSVs include `low` and `high` interval bounds; negative differences and intervals are retained.

Random uses exactly one allocation per instance and budget: each sensor is independently assigned to a uniformly chosen location, matching Problem 2’s sampling procedure. A deterministic seed makes the draw reproducible, and the same allocation is evaluated for all five probes. There is no within-instance trial averaging.
