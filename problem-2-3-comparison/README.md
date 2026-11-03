# Sequential versus joint discrimination

Companion notebook for paper §6: sequential Heuristic (VND + Beam) versus joint Greedy with the local equal-Dicke-sector state and PGM.

Use the [root setup](../README.md) and open `01_sequential_vs_joint_analysis.ipynb`. Keep `comparison/` and the other problem directories: the notebook imports their implementations.

- `RUN_MODE`: `"full"` or `"compact"` (`"test"` is an alias).
- `SEQUENTIAL_ALGORITHM` / `JOINT_ALGORITHM`: select the methods independently.
- `RUN_ANALYSIS=False`: require saved checkpoints; `True` computes missing evaluations.

Inputs: `data/`. Outputs: `results/{full,compact}/`; reports overwrite existing exports, while per-algorithm checkpoints remain reusable. Retain the caches.

The figure uses matched target-fraction groups; the family table weights instances equally after averaging unique budgets. The final sensor-use table averages unique instance–budget pairs separately for 1D/2D, using saved sequential trees and joint uses equal to N.
