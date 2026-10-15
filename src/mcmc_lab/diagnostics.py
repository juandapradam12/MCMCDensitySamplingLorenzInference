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


def _integrated_autocorr_time(x: np.ndarray) -> float:
    """Initial positive sequence estimate of integrated autocorrelation time."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return 1.0

    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0.0:
        return 1.0

    # Biased autocovariance via FFT for O(n log n) cost.
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n].real / n

    max_lag = n // 2
    tau = 1.0
    for lag in range(1, max_lag):
        rho = acov[lag] / var
        if rho < 0.0:
            break
        tau += 2.0 * rho
    return max(tau, 1.0)


def effective_sample_size(chains: list[ChainResult] | np.ndarray) -> np.ndarray:
    """Effective sample size per parameter.

    Accepts either a list of ``ChainResult`` (ESS summed across chains) or a
    single ``(n_draws, n_params)`` array.
    """
    if isinstance(chains, np.ndarray):
        draws = np.asarray(chains, dtype=float)
        if draws.ndim == 1:
            draws = draws[:, None]
        return np.array(
            [draws.shape[0] / _integrated_autocorr_time(draws[:, j]) for j in range(draws.shape[1])],
            dtype=float,
        )

    if not chains:
        raise ValueError("effective_sample_size requires at least one chain")

    n_params = chains[0].posterior.shape[1]
    ess = np.zeros(n_params, dtype=float)
    for chain in chains:
        posterior = chain.posterior
        for j in range(n_params):
            ess[j] += posterior.shape[0] / _integrated_autocorr_time(posterior[:, j])
    return ess
