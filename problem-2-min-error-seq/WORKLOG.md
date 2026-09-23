# Migration status

- Independent copy created; original code, notebooks, and inputs verified unchanged.
- Exhaustive search and Greedy use actual-error objectives.
- Heuristic scores guide construction; all methods report actual tree error.
- Old analyses and saved results removed from the copy.
- Compact, extended, and final notebooks replaced with actual-error comparisons.
- Appendix D closed-form off-pair probabilities now drive every reported error.
- Input CSVs contain only `i,j,m,r_ijm`; all four datasets were regenerated.
- Test runs select 180 cases per spatial dimension, balanced across field families.
- Greedy tree construction now uses only local `P_ijmn`; exact `P`/`Q` error
  remains the outer sensor-placement objective and final reported value.
- Exhaustive search was checked against the new Section 6 definition and already
  enumerated all allocations and node-specific pair/location choices exactly.
- Thirteen unit tests and both 360-case command-line studies passed after the update.
- Checkpoint reuse, figures, generators, and small K=4 searches checked.
- Full studies have not been run. Start with RUN_MODE="test"; exact full runs default to K≤4.
