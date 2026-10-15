"""Visualization helpers for sampler validation and inference results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from mcmc_lab.diagnostics import pooled_posterior
from mcmc_lab.lorenz import LorenzParameters, integrate_lorenz
from mcmc_lab.metropolis import ChainResult


def plot_normal_sampling(
    chains: list[np.ndarray],
    mu: float = 0.0,
    sigma: float = 1.0,
    output_path: str | Path = "outputs/normal_sampling.png",
) -> Path:
    """Overlay chain histograms against the analytical Normal density."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for samples in chains:
        ax.hist(samples, bins=40, density=True, alpha=0.35, edgecolor="none")
    ax.plot(x, norm.pdf(x, loc=mu, scale=sigma), color="#1b4332", lw=2.5, label="N(0,1)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Metropolis-Hastings sampling of N(0, 1)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_lorenz_inference(
    chains: list[ChainResult],
    times: np.ndarray,
    observations: np.ndarray,
    output_prefix: str | Path = "outputs/lorenz",
) -> dict[str, Path]:
    """Save trace, posterior marginals, and predictive trajectory plots."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    labels = [r"$\sigma$", r"$\rho$", r"$\beta$"]
    names = ["sigma", "rho", "beta"]
    pooled = pooled_posterior(chains)

    # Trace plots
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, ax in enumerate(axes):
        for chain in chains:
            ax.plot(chain.samples[:, i], lw=0.6, alpha=0.8)
        ax.axvline(chain.n_burnin, color="#6c757d", ls="--", lw=1)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("iteration")
    fig.suptitle("Parameter traces")
    fig.tight_layout()
    paths["traces"] = prefix.with_name(prefix.name + "_traces.png")
    fig.savefig(paths["traces"], dpi=150)
    plt.close(fig)

    # Marginal posteriors
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for i, ax in enumerate(axes):
        ax.hist(pooled[:, i], bins=40, density=True, color="#2d6a4f", alpha=0.85)
        ax.set_xlabel(labels[i])
        ax.set_title(names[i])
    axes[0].set_ylabel("density")
    fig.suptitle("Posterior marginals")
    fig.tight_layout()
    paths["marginals"] = prefix.with_name(prefix.name + "_marginals.png")
    fig.savefig(paths["marginals"], dpi=150)
    plt.close(fig)

    # Predictive trajectory using posterior mean
    mean_params = LorenzParameters.from_array(pooled.mean(axis=0))
    predicted = integrate_lorenz(mean_params, observations[0], times)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    coord_labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(times, observations[:, i], "o", ms=4, color="#1d3557", label="observed")
        ax.plot(times, predicted[:, i], color="#e76f51", lw=2, label="posterior mean")
        ax.set_ylabel(coord_labels[i])
    axes[-1].set_xlabel("t")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Lorenz trajectory fit")
    fig.tight_layout()
    paths["fit"] = prefix.with_name(prefix.name + "_fit.png")
    fig.savefig(paths["fit"], dpi=150)
    plt.close(fig)

    return paths
