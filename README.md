# Quantum Sensor Placement Research

Companion code for *Sensor Allocation for Spatial Field Discrimination Using Quantum Sensor Networks*. See the paper for formulations, algorithms, derivations and experimental design.

| Study | Paper | Notebook directory |
| --- | --- | --- |
| Perfect discrimination | §§3.4–3.5 | [Problem 1](problem-1-zero-error/README.md) |
| Sequential minimum error | §4.4 | [Problem 2](problem-2-min-error-seq/README.md) |
| Target-error sensor budgets | §4.5 | [Problem 2 dual](problem-2-min-error-seq/problem-2-dual/README.md) |
| Single-location optimization | §5.1 | [Single location](problem-3-single-location/README.md) |
| Joint minimum error | §§5.2–5.3 | [Problem 3](problem-3-global-pgm/README.md) |
| Sequential versus joint | §6 | [Comparison](problem-2-3-comparison/README.md) |

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problem-1-zero-error/requirements.txt -r problem-2-min-error-seq/requirements.txt -r problem-3-global-pgm/requirements.txt -r problem-3-single-location/requirements.txt -r problem-2-3-comparison/requirements.txt
python -m jupyterlab
```

Keep the repository structure intact: studies share implementation modules. Select the notebook's mode and execution settings before running cells in order. Use a fresh kernel when switching studies. `final_figures.ipynb` notebooks read saved full results.

Generated inputs and results are local artifacts, excluded from Git. Retain them and their checkpoints to reproduce saved figures without rerunning experiments.

License: [Apache 2.0](LICENSE).
