"""Metropolis-Hastings MCMC lab: sampling and Bayesian dynamical-system inference."""

from mcmc_lab.diagnostics import effective_sample_size, gelman_rubin, summarize_chain
from mcmc_lab.metropolis import MetropolisHastings, ChainResult
from mcmc_lab.lorenz import LorenzParameters, integrate_lorenz

__all__ = [
    "MetropolisHastings",
    "ChainResult",
    "LorenzParameters",
    "integrate_lorenz",
    "gelman_rubin",
    "effective_sample_size",
    "summarize_chain",
]

__version__ = "1.1.0"
