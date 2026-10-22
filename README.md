# Quantum Sensor Placement Research

Code and datasets for *Sensor Allocation for Spatial Field Discrimination Using
Quantum Sensor Networks*.

| Folder | Paper sections | Study |
| --- | --- | --- |
| [problem-1-zero-error](problem-1-zero-error/README.md) | 3.7–3.8 | Sequential perfect discrimination and the joint sensor lower bound |
| [problem-2-min-error-seq](problem-2-min-error-seq/README.md) | 4.6 | Sequential minimum-error discrimination |
| [problem-2-dual](problem-2-min-error-seq/problem-2-dual/README.md) | 4.7 | Sensor budgets for a target error |
| [problem-3-single-location](problem-3-single-location/README.md) | 5.2 | Single-location input-state and measurement optimization |
| [problem-3-global-pgm](problem-3-global-pgm/README.md) | 5.5 | Multi-location input states and sensor allocations with PGM |
| [problem-2-3-comparison](problem-2-3-comparison/README.md) | 6 | Matched sequential and joint discrimination |

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problem-1-zero-error/requirements.txt -r problem-2-min-error-seq/requirements.txt -r problem-3-global-pgm/requirements.txt -r problem-3-single-location/requirements.txt -r problem-2-3-comparison/requirements.txt
python -m jupyterlab
```

License: [Apache 2.0](LICENSE).
