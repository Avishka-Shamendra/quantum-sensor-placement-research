# Zero-Error State Overlaps

Run `zero_error_overlap_sanity_check.ipynb` to check saved phase-gap inputs.
For each nonzero gap, the sensor threshold is `ceil(pi / abs(r_ijm))`.
The overlap is evaluated from sparse eigen-components for the boundary,
even-threshold, or odd-threshold construction, without building a full state vector.

Use `RUN_SCOPE = "test"` or `"full"`. Results go to `outputs/`:

- `overlap_summary.csv`: aggregate residuals and violation counts.
- `overlap_violations.csv`: overlaps exceeding the tolerance.
- `overlap_sample.csv`: sample checks.
