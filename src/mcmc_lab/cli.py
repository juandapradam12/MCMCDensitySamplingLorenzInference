"""Command-line entry points for the two portfolio use cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcmc_lab.diagnostics import effective_sample_size, gelman_rubin, pooled_posterior
from mcmc_lab.lorenz import load_observations, make_lorenz_log_posterior
from mcmc_lab.metropolis import MetropolisHastings
from mcmc_lab.plotting import plot_lorenz_inference, plot_normal_sampling


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sample_normal_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Metropolis-Hastings by sampling N(0, 1)."
    )
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "outputs" / "normal_sampling.png",
    )
    args = parser.parse_args(argv)

    def log_prob(theta: np.ndarray) -> float:
        x = float(theta[0])
        return -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)

    rng = np.random.default_rng(args.seed)
    initials = [np.array([rng.normal()]) for _ in range(args.n_chains)]
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(args.n_chains)]

    sampler = MetropolisHastings(log_prob=log_prob, step_size=args.step_size, seed=args.seed)
    chains = sampler.run_chains(initials, n_steps=args.n_steps, burnin=args.burnin, seeds=seeds)

    posteriors = [c.posterior[:, 0] for c in chains]
    path = plot_normal_sampling(posteriors, output_path=args.output)
    rhat = gelman_rubin(chains)
    pooled = pooled_posterior(chains)

    ess = effective_sample_size(chains)
    print(f"chains           : {args.n_chains}")
    print(f"acceptance rates : {[round(c.acceptance_rate, 3) for c in chains]}")
    print(f"R-hat            : {rhat.round(4)}")
    print(f"ESS              : {ess.round(1)}")
    print(f"posterior mean   : {pooled.mean(axis=0).round(4)}")
    print(f"figure           : {path}")
    return 0


def infer_lorenz_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Infer Lorenz-63 parameters from observational trajectory data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_repo_root() / "data" / "lorenz_observations.dat",
    )
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--n-chains", type=int, default=3)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.12)
    parser.add_argument("--sigma-obs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_repo_root() / "outputs" / "lorenz",
    )
    args = parser.parse_args(argv)

    times, observations = load_observations(str(args.data))
    log_posterior = make_lorenz_log_posterior(
        times, observations, sigma_obs=args.sigma_obs
    )

    rng = np.random.default_rng(args.seed)
    # Dispersed starts spanning a broad but physically plausible Lorenz regime.
    centers = [
        np.array([10.0, 28.0, 8.0 / 3.0]),
        np.array([9.0, 20.0, 4.0]),
        np.array([12.0, 24.0, 2.5]),
    ]
    initials = []
    for i in range(args.n_chains):
        base = centers[i % len(centers)]
        initials.append(np.clip(base + rng.normal(0.0, 1.5, size=3), 0.5, 39.5))
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(args.n_chains)]

    sampler = MetropolisHastings(
        log_prob=log_posterior, step_size=args.step_size, seed=args.seed
    )
    chains = sampler.run_chains(
        initials, n_steps=args.n_steps, burnin=args.burnin, seeds=seeds
    )

    pooled = pooled_posterior(chains)
    rhat = gelman_rubin(chains)
    ess = effective_sample_size(chains)
    paths = plot_lorenz_inference(
        chains, times, observations, output_prefix=args.output_prefix
    )

    labels = ["sigma", "rho", "beta"]
    print("Lorenz-63 parameter inference")
    print(f"acceptance rates : {[round(c.acceptance_rate, 3) for c in chains]}")
    print(f"R-hat            : {dict(zip(labels, np.round(rhat, 4)))}")
    print(f"ESS              : {dict(zip(labels, np.round(ess, 1)))}")
    print(f"posterior mean   : {dict(zip(labels, np.round(pooled.mean(axis=0), 4)))}")
    print(
        "95% CI           : "
        f"{dict(zip(labels, zip(np.round(np.quantile(pooled, 0.025, axis=0), 4), np.round(np.quantile(pooled, 0.975, axis=0), 4))))}"
    )
    for name, path in paths.items():
        print(f"{name:16s} : {path}")
    return 0
