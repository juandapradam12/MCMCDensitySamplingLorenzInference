# Use Case

## Problem

Given a short trajectory $(t, x, y, z)$ generated from Lorenz-63 with known parameters and observation noise, recover $(\sigma, \rho, \beta)$ under uncertainty. Before trusting that inference pipeline, the Metropolis-Hastings engine is validated on a known target: the unit Normal distribution.

## Why this matters

Dynamical systems with chaotic sensitivity make point estimates brittle. A Bayesian MCMC workflow returns posterior uncertainty, acceptance diagnostics, multi-chain convergence checks (Gelman-Rubin $\hat{R}$), and effective sample size (ESS)—the quantities a practitioner needs to decide whether the fit is usable.

## Approach

1. **Sampler validation** — draw multiple chains from $N(0,1)$, compare histograms to the analytical density, and inspect acceptance rates / $\hat{R}$.
2. **Forward model** — integrate Lorenz-63 with candidate parameters from the observed initial state.
3. **Likelihood** — Gaussian residual model between simulated and observed states.
4. **Prior** — independent box constraints on each parameter.
5. **Inference** — multi-chain Metropolis-Hastings; report posterior means, credible intervals, $\hat{R}$, ESS, traces, and predictive trajectory overlays.

## Scope

This repository packages a focused computational statistics case study: one sampler, one dynamical system, and the diagnostics required to present the result as a credible portfolio artifact—not a general-purpose probabilistic programming framework.
