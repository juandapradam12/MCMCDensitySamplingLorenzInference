"""Lorenz-63 dynamical system and Bayesian observation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import odeint


@dataclass(frozen=True)
class LorenzParameters:
    sigma: float
    rho: float
    beta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.sigma, self.rho, self.beta], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "LorenzParameters":
        return cls(sigma=float(values[0]), rho=float(values[1]), beta=float(values[2]))


def lorenz_rhs(state: np.ndarray, _t: float, params: LorenzParameters) -> list[float]:
    x, y, z = state
    dx = params.sigma * (y - x)
    dy = x * (params.rho - z) - y
    dz = x * y - params.beta * z
    return [dx, dy, dz]


def integrate_lorenz(
    params: LorenzParameters,
    initial_state: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Integrate Lorenz-63 and return states shaped (n_times, 3)."""
    trajectory = odeint(
        lorenz_rhs,
        np.asarray(initial_state, dtype=float),
        np.asarray(times, dtype=float),
        args=(params,),
        rtol=1e-6,
        atol=1e-8,
    )
    return trajectory


def load_observations(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load observational file with columns: t, x, y, z."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("Expected observational data with columns t, x, y, z")
    times = data[:, 0]
    states = data[:, 1:4]
    return times, states


def make_lorenz_log_posterior(
    times: np.ndarray,
    observations: np.ndarray,
    sigma_obs: float = 1.0,
    prior_bounds: tuple[float, float] = (0.0, 40.0),
) -> Callable[[np.ndarray], float]:
    """Gaussian likelihood on trajectory residuals with box prior on parameters."""

    initial_state = observations[0].copy()
    lower, upper = prior_bounds
    inv_var = 1.0 / (sigma_obs ** 2)
    n_obs = observations.size

    def log_prior(theta: np.ndarray) -> float:
        if np.any(theta <= lower) or np.any(theta >= upper):
            return -np.inf
        return 0.0

    def log_likelihood(theta: np.ndarray) -> float:
        params = LorenzParameters.from_array(theta)
        try:
            predicted = integrate_lorenz(params, initial_state, times)
        except Exception:
            return -np.inf
        residual = observations - predicted
        return -0.5 * inv_var * float(np.sum(residual ** 2)) - 0.5 * n_obs * np.log(
            2.0 * np.pi * sigma_obs ** 2
        )

    def log_posterior(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    return log_posterior
