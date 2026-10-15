"""Metropolis-Hastings sampler with multi-chain support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


LogProbFn = Callable[[np.ndarray], float]
ProposalFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class ChainResult:
    """Posterior samples and basic run metadata for one Markov chain."""

    samples: np.ndarray
    log_prob: np.ndarray
    acceptance_rate: float
    n_burnin: int

    @property
    def posterior(self) -> np.ndarray:
        """Samples after burn-in."""
        return self.samples[self.n_burnin :]


def _default_gaussian_proposal(
    step_size: float,
) -> ProposalFn:
    def proposal(current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return current + rng.normal(0.0, step_size, size=current.shape)

    return proposal


class MetropolisHastings:
    """Random-walk Metropolis-Hastings for continuous parameter vectors.

    Independent chains can be launched via ``run_chains``. Updates inside a
    single chain stay sequential, as required by the Markov property.
    """

    def __init__(
        self,
        log_prob: LogProbFn,
        proposal: ProposalFn | None = None,
        step_size: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.log_prob = log_prob
        self.step_size = float(step_size)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.proposal = proposal or _default_gaussian_proposal(self.step_size)

    def run(
        self,
        initial: np.ndarray,
        n_steps: int,
        burnin: int = 0,
    ) -> ChainResult:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if burnin < 0 or burnin >= n_steps:
            raise ValueError("burnin must satisfy 0 <= burnin < n_steps")

        current = np.asarray(initial, dtype=float).copy()
        current_lp = float(self.log_prob(current))

        samples = np.empty((n_steps, current.size), dtype=float)
        log_probs = np.empty(n_steps, dtype=float)
        accepted = 0

        for i in range(n_steps):
            proposed = self.proposal(current, self._rng)
            proposed_lp = float(self.log_prob(proposed))
            log_alpha = proposed_lp - current_lp

            if np.log(self._rng.random()) < log_alpha:
                current = proposed
                current_lp = proposed_lp
                accepted += 1

            samples[i] = current
            log_probs[i] = current_lp

        return ChainResult(
            samples=samples,
            log_prob=log_probs,
            acceptance_rate=accepted / n_steps,
            n_burnin=burnin,
        )

    def run_chains(
        self,
        initials: list[np.ndarray] | np.ndarray,
        n_steps: int,
        burnin: int = 0,
        seeds: list[int] | None = None,
    ) -> list[ChainResult]:
        """Run several independent chains with optional per-chain seeds."""
        starts = [np.asarray(x, dtype=float) for x in initials]
        if seeds is not None and len(seeds) != len(starts):
            raise ValueError("seeds length must match number of chains")

        results: list[ChainResult] = []
        for idx, start in enumerate(starts):
            if seeds is None:
                chain_seed = int(self._rng.integers(0, 2**31 - 1))
            else:
                chain_seed = seeds[idx]

            sampler = MetropolisHastings(
                log_prob=self.log_prob,
                proposal=self.proposal,
                step_size=self.step_size,
                seed=chain_seed,
            )
            results.append(sampler.run(start, n_steps=n_steps, burnin=burnin))

        return results
