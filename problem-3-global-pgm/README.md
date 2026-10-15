# Problem 3: allocation with global PGM evaluation

`01_compact_analysis.ipynb` compares Exhaustive, Greedy and VND. `02_extended_analysis.ipynb` compares Greedy and VND without Exhaustive. Compact uses `inputs/compact`; Extended uses `inputs/extended`. Both support test/full mode and resumable matching results. Neither notebook is executed during preparation.

## Algorithms

For every field pair, d_ijm is the absolute actual encoded phase difference. Greedy starts at zero, selects the currently least-separated pair, and adds a sensor at that pair's strongest location. Tied pairs prefer the smaller maximum separation, then lexicographic pair order; tied locations prefer the smallest index.

VND independently starts near uniformly, assigning remainders to the first locations. It maximizes S(n) = mean_ij min(pi, sum_m n_m d_ijm), scanning best-improvement transfers of one, two, and three sensors. Improvements greater than 1e-12 restart the one-sensor neighborhood. Source/destination iteration order breaks ties deterministically. VND stops at convergence; the prior 30-iteration cap is superseded by this specification. A timeout is reported as incomplete.

Neither heuristic evaluates PGM during search. One allocation per instance and budget is applied to all five unchanged probe families. Their final actual global PGM errors are evaluated separately. Improving the phase score does not guarantee improving PGM error.

Exhaustive enumerates every feasible allocation separately for each probe state, retaining the first minimum in allocation enumeration order. It is optimal over allocations for that state and PGM measurement, not over all quantum measurements.

## Results

Exactly four figures are produced: Compact state curves using Exhaustive and heuristic gap bars; Extended state curves using VND and grouped error-difference bars relative to VND. Each figure has 1D/2D panels, 95% confidence intervals, Times New Roman 12 pt text, PDF and PNG exports. Overall and field-family tables include runtime, error, Compact optimality-match and VND win/tie/loss summaries. Random and Near-Uniform references are enabled by default. Both second figures are grouped bars by input state, with 95% confidence intervals; negative Extended differences indicate lower error than VND.

The dataset generator and validation commands are documented in [benchmark_pipeline/README.md](benchmark_pipeline/README.md). Cached results remain separate; source, settings and dataset fingerprints prevent stale-result reuse.

Extended uses the exact Problem 2 Extended CSV instances: K=3–10, M=2–10, 10 replicates per family/configuration/dimension, 4,320 total. Problem 3 retains joint-lower-bound budgets and normalization. Compact retains K,M=3–5 and 50 replicates per cell and 2,700 instances. The positive 0.01 gap filter is removed; exactly indistinguishable pairs remain invalid for a finite joint bound.
