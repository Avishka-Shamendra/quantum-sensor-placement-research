# Shared Minimum-Error Datasets

This directory owns the synthetic input generator, sparse CSV loader, and the
four benchmark designs shared by minimum-error sequential and global
discrimination.

The Gaussian, sinusoidal, and quadratic field families and their parameter
ranges are the definitions from Tables I and II of the paper. Generated CSVs
store canonical phase gaps `r_ijm` and sparse pairwise error values. Sequential
algorithms use both; global algorithms use the phase gaps.

## Field Families

For 1D locations `alpha` in `[0, 1]`:

| Family | Function | Parameter ranges |
| --- | --- | --- |
| Gaussian | `f_k(alpha) = A_k exp(-(alpha-mu_k)^2/(2w_k^2))` | `A_k in [0.5,1.5]`, `mu_k in [0,1]`, `w_k in [0.1,0.3]` |
| Sinusoidal | `f_k(alpha) = A_k sin(2 pi q_k alpha + gamma_k)` | `A_k in [0.5,1.5]`, `q_k in {1,2,3}`, `gamma_k in [-pi,pi]` |
| Quadratic | `f_k(alpha) = b_k + a_k(alpha-mu_k)^2` | `b_k in [-0.5,0.5]`, `a_k in [-1,1]`, `mu_k in [0,1]` |

For 2D locations `(alpha, beta)` in `[0, 1]^2`:

| Family | Function | Parameter ranges |
| --- | --- | --- |
| Gaussian | `f_k(alpha,beta) = A_k exp(-((alpha-mu_k1)^2+(beta-mu_k2)^2)/(2w_k^2))` | `A_k in [0.5,1.5]`, `mu_k1,mu_k2 in [0,1]`, `w_k in [0.1,0.3]` |
| Sinusoidal | `f_k(alpha,beta) = A_k sin(2 pi q_k(alpha cos(theta_k)+beta sin(theta_k))+gamma_k)` | `A_k in [0.5,1.5]`, `q_k in {1,2,3}`, `theta_k in [0,pi)`, `gamma_k in [-pi,pi]` |
| Quadratic | `f_k(alpha,beta) = b_k + (a_k/2)((alpha-mu_k1)^2+(beta-mu_k2)^2)` | `b_k in [-0.5,0.5]`, `a_k in [-1,1]`, `mu_k1,mu_k2 in [0,1]` |

## Benchmark Designs

| Dataset | K | M | Cases |
| --- | --- | --- | ---: |
| `compact_benchmark_1d` | 3-5 | 3-4 | 2,520 |
| `compact_benchmark_2d` | 3-5 | 3-4 | 2,520 |
| `benchmark_1d` | 3-7 | 3-8 | 2,700 |
| `benchmark_2d` | 3-7 | 3-8 | 2,700 |

Generate the compact datasets:

```bash
python min-error-datasets/generate_compact_benchmarks.py all
```

Generate the extended datasets:

```bash
python min-error-datasets/generate_benchmarks.py all
```

The committed generator settings and fixed seeds reproduce the existing data
without changing its design or numerical values.
