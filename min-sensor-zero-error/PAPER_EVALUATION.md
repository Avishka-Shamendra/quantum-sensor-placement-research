# Evaluation Strategy for the Paper

## Research decision

The algorithm-comparison direction is sound, but 20 heterogeneous cases are
only a software smoke test. They do not provide enough replication within a
configuration to estimate variability, check robustness, or support a paper
claim.

The revised evaluation separates three questions:

1. **Primary performance:** how close is greedy to the exact optimum over a
   balanced synthetic problem distribution?
2. **Reproducibility:** does that conclusion persist under an independent seed?
3. **Robustness:** does changing the phase scale materially change the greedy
   excess?

The experimental unit is one independently generated set of `K` candidate
fields, observed at `M` fixed sensing locations. The exact and greedy methods
are run on the same unit, so every quality comparison is paired by
construction.

## Pre-specified datasets

| Dataset | Models | `K` | `M` | `c` | Realizations per cell | Total |
|---|---:|---:|---:|---:|---:|---:|
| Primary | 4 | 3–10 | 3, 5, 8 | 1 | 30 | 2,880 |
| Independent replication | 4 | 3–10 | 3, 5, 8 | 1 | 10 | 960 |
| Paired sensitivity | 4 | 4, 7, 10 | 3, 8 | 0.5, 1, 2 | 15 | 1,080 |

The four model families—Gaussian, sine, Fourier, and polynomial—represent
different smooth spatial structures. Treat them as a deliberately varied
synthetic test distribution, not as a standard external benchmark.

The fixed designs and seeds are defined in `generate_paper_studies.py`; all
realized parameters are recorded in manifests. This prevents accidental
selection of only favorable cases after seeing the results.

## Why 30 realizations per cell

There is no useful classical power calculation against the approximation
guarantee: the guarantee is a deterministic upper bound, not a stochastic null
hypothesis. The relevant question is Monte Carlo precision.

Thirty realizations in each of 96 primary factorial cells give:

- 2,880 independent primary cases;
- 360 cases at every value of `K`;
- 720 cases per field family; and
- balanced coverage of all model × `K` × `M` combinations.

The convergence diagnostic recomputes the pooled mean ratio, 95th percentile,
and exact-match rate after 5, 10, 15, 20, 25, and 30 realizations per cell. In
the current data, increasing from 25 to 30 changed the mean ratio by 0.0001 and
the exact-match estimate by 0.28 percentage points. This supports 30 for the
pooled and main-factor summaries. It does not make every individual 30-case
factorial cell precise enough for a standalone claim.

## Outcomes

Use the greedy-to-optimal sensor ratio as the primary outcome:

```text
rho = N_greedy / N_optimal.
```

Report all of the following:

- mean ratio with a 95% factorial-cell-stratified bootstrap interval;
- median, 95th percentile, and maximum ratio;
- exact-optimum recovery rate with a 95% Wilson interval;
- percentage additional sensors used by greedy;
- observed fraction of the theorem's permitted excess; and
- median runtimes with interquartile ranges on a log scale.

For a problem with `K` fields, the worst-case factor is

```text
B(K) = 1 + ln(K(K - 1) / 2).
```

The clean comparison with this guarantee is

```text
(rho - 1) / (B(K) - 1),
```

not `rho / B(K)`. The former asks what fraction of the *additional sensors*
permitted by the guarantee was actually used; it is zero when greedy is exact.

Bootstrap resampling is performed within factorial cells so that uncertainty
estimation preserves the balanced design. Phase-scale intervals use paired
differences and resample within model × `M` strata.

## Figure plan

### Main text: solution quality

Use `primary_quality.pdf` as the main empirical figure:

- panel (a): distribution of percentage sensor excess by `K`;
- panel (b): exact-optimum recovery with Wilson intervals; and
- panel (c): observed 95th percentile and maximum as fractions of the
  worst-case permitted excess.

This figure directly answers the main research question without showing
thousands of unreadable case labels.

### Main text or systems section: runtime

Use `primary_runtime_scaling.pdf`. It shows medians and interquartile bands for
each `M` on a log time scale. Before using it in the manuscript, rerun with at
least five timing repetitions on an idle machine and report processor,
operating system, Python version, and repetition count.

### Supplement or robustness section

- `primary_quality_ecdf.pdf`: full quality distribution by field family;
- `phase_scale_sensitivity.pdf`: paired changes from `c = 1` with intervals;
- `primary_sample_convergence.pdf`: evidence for Monte Carlo stability; and
- `group_summary.csv`: numerical results by model, `K`, and `M`.

## Defensible result language

Use the automatically refreshed wording in `output/paper-analysis/paper_findings.md`.
The intended conclusion is:

> On the pre-specified synthetic spatial-field distribution, greedy was
> empirically much closer to the exact optimum than the worst-case logarithmic
> allowance, and the aggregate result reproduced under an independent seed.

Do not write that the experiments improve the approximation guarantee. Do not
describe the synthetic results alone as proof that greedy performs this well
in every practical deployment.

## What is still needed for a strong practical claim

The current evaluation is a solid algorithmic synthetic study. A claim using
the phrase “in practice” would be stronger with at least one application-based
dataset, for example measured fields, physics-simulator outputs calibrated to
the intended sensor, or parameter ranges justified from domain literature.
That dataset should be kept separate from the synthetic primary study and
reported as an external-validity experiment.

Before submission:

1. justify the chosen field families and parameter ranges from the application;
2. add an application-realistic or measured dataset if one is available;
3. rerun timings with repeated solves and record hardware/software details;
4. use the independent replication as confirmation, not for retuning choices;
5. label any later design changes made after viewing these results as
   exploratory; and
6. state that exact comparison is restricted to `K <= 10` by the exponential
   reference solver.
