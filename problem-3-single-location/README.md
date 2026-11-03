# Problem 3: single-location optimization

See paper §5.1. Dicke-sector weights and the POVM are optimized jointly by SDP, with numerical feasibility and optimality checks. Reflection symmetry is not imposed.

Use the [root setup](../README.md) and open `01_single_location_analysis.ipynb`. Generate missing inputs with `python problem-3-single-location/scripts/generate_datasets.py all` from the repository root.

`RUN_ANALYSIS=False` loads saved results; `True` evaluates missing cases using compatible checkpoints. Inputs: `inputs/{1d,2d}/`. Results, figures and checkpoints: `results/full/`.
