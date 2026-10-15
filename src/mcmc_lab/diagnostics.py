"""Convergence and summary diagnostics for MCMC chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc_lab.metropolis import ChainResult


@dataclass(frozen=True)
class ParameterSummary:
    mean: np.ndarray
    std: np.ndarray
    q025: np.ndarray
    q975: np.ndarray
    acceptance_rate: float


def summarize_chain(chain: ChainResult) -> ParameterSummary:
    posterior = chain.posterior
    return ParameterSummary(
        mean=posterior.mean(axis=0),
        std=posterior.std(axis=0, ddof=1),
        q025=np.quantile(posterior, 0.025, axis=0),
        q975=np.quantile(posterior, 0.975, axis=0),
        acceptance_rate=chain.acceptance_rate,
    )


def gelman_rubin(chains: list[ChainResult]) -> np.ndarray:
    """Potential scale reduction factor (R-hat) per parameter.

    Uses the classic Gelman-Rubin estimator on post-burn-in samples.
    Values near 1.0 indicate approximate convergence across chains.
    """
    if len(chains) < 2:
        raise ValueError("gelman_rubin requires at least two chains")

    draws = [c.posterior for c in chains]
    n = min(d.shape[0] for d in draws)
    if n < 2:
        raise ValueError("Each chain needs at least 2 post-burn-in samples")

    stacked = np.stack([d[-n:] for d in draws], axis=0)  # (m, n, p)
    m, _, p = stacked.shape

    chain_means = stacked.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)

    b = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    w = stacked.var(axis=1, ddof=1).mean(axis=0)

    var_hat = ((n - 1) / n) * w + b / n
    r_hat = np.sqrt(var_hat / w)
    return r_hat


def pooled_posterior(chains: list[ChainResult]) -> np.ndarray:
    """Concatenate post-burn-in samples across chains."""
    return np.concatenate([c.posterior for c in chains], axis=0)
