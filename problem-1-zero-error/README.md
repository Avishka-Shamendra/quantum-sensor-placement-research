# Problem 1: perfect discrimination

See paper §§3.4–3.5 for sequential DP, Greedy and the joint sensor lower bound.

- `01_solution_method_analysis.ipynb`: DP/Greedy comparison.
- `02_minimum_sensor_trends.ipynb`: sequential minimum sensor budgets.
- `03_global_seq_comparison.ipynb`: sequential budgets versus the joint lower bound.
- `final_figures.ipynb`: publication figures from saved full results.

Use the [root setup](../README.md). Generate missing inputs with `python problem-1-zero-error/scripts/generate_benchmarks.py all` from the repository root. Notebooks 01–03 run independently with `RUN_MODE="test"` or `"full"`; 02–03 reuse compatible DP results when `REUSE_SAVED_RESULTS=True`.

Results: `results/{methods,nmin,global}/{test,full}/`. Final figures require all three full studies. The joint integer-program bound does not itself certify an achievable joint input state.
