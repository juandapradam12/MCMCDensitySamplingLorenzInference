# Results Snapshot

Reproducible summaries from the default CLI seeds. Regenerate with:

```bash
python -m mcmc_lab sample-normal
python -m mcmc_lab infer-lorenz
```

## Sampler validation — $N(0,1)$

| Metric | Value |
| --- | --- |
| Chains × steps (burn-in) | 4 × 5000 (1000) |
| Acceptance rates | 0.70, 0.71, 0.71, 0.70 |
| $\hat{R}$ | 1.0006 |
| ESS | ≈ 2034 |
| Posterior mean | −0.024 |

Histograms of the post-burn-in draws match the analytical unit Normal density (`docs/figures/normal_sampling.png`).

## Lorenz-63 parameter inference

Ground truth:

$$
(\sigma, \rho, \beta) = (10, 28, 8/3).
$$

| Parameter | Posterior mean | 95% CI | $\hat{R}$ | ESS |
| --- | --- | --- | --- | --- |
| $\sigma$ | 9.92 | [9.60, 10.21] | 1.018 | ≈ 134 |
| $\rho$ | 27.92 | [27.62, 28.25] | 1.025 | ≈ 121 |
| $\beta$ | 2.59 | [2.53, 2.66] | 1.000 | ≈ 574 |

Acceptance rates ≈ 0.21–0.23. Predictive trajectory from the posterior mean overlays the noisy observations (`docs/figures/lorenz_fit.png`).

## C backend check

```bash
make c-sampler
python -m mcmc_lab sample-normal-c 2000 4
```

Example output: `shape=(4, 2000) mean≈0.00 std≈1.01`.
