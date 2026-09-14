# Zero-Error Appendix Overlap Sanity Check

This folder checks the quantum-state construction used for the zero-error
sensor thresholds.

For each saved input case in `benchmark_1d` and `benchmark_2d`, the checker:

1. Loads every phase gap `r_ijm`.
2. Computes `N_ijm = ceil(pi / |r_ijm|)` for nonzero gaps.
3. Applies the appropriate appendix construction:
   - boundary case, `N_ijm |r_ijm| = pi`;
   - above-threshold case with even `N_ijm`;
   - above-threshold case with odd `N_ijm`.
4. Evaluates `<psi|V_ijm^{tensor N_ijm}|psi>` using the eigen-component
   representation of the appendix state.
5. Reports the absolute overlap and flags any value above the configured
   numerical tolerance.

The implementation does not build full `2^N_ijm` state vectors. Some benchmark
phase gaps are very small, so `N_ijm` can be extremely large. Instead, the code
stores only the nonzero eigen-components of the appendix state, which is
mathematically equivalent for this overlap calculation.

Run `zero_error_overlap_sanity_check.ipynb` to produce:

- `outputs/overlap_summary.csv`
- `outputs/overlap_violations.csv`
- `outputs/overlap_sample.csv`
