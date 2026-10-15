#!/usr/bin/env python3
"""Regenerate the synthetic Lorenz observations used by the inference demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcmc_lab.lorenz import LorenzParameters, integrate_lorenz


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "lorenz_observations.dat"

    params = LorenzParameters(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    times = np.linspace(0.0, 1.2, 25)
    x0 = np.array([1.0, 1.0, 1.0])
    truth = integrate_lorenz(params, x0, times)

    rng = np.random.default_rng(2026)
    obs = truth + rng.normal(0.0, 0.75, size=truth.shape)
    obs[0] = truth[0]

    np.savetxt(out, np.column_stack([times, obs]), fmt="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
