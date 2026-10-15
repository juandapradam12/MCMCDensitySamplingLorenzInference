import numpy as np

from mcmc_lab.diagnostics import gelman_rubin, pooled_posterior
from mcmc_lab.metropolis import MetropolisHastings


def _normal_log_prob(theta: np.ndarray) -> float:
    x = float(theta[0])
    return -0.5 * x * x


def test_metropolis_recovers_standard_normal_mean():
    sampler = MetropolisHastings(log_prob=_normal_log_prob, step_size=1.0, seed=123)
    chains = sampler.run_chains(
        initials=[np.array([x]) for x in (-2.0, -1.0, 1.0, 2.0)],
        n_steps=3000,
        burnin=800,
        seeds=[1, 2, 3, 4],
    )

    pooled = pooled_posterior(chains)[:, 0]
    assert abs(pooled.mean()) < 0.15
    assert 0.7 < pooled.std(ddof=1) < 1.3
    assert all(0.1 < c.acceptance_rate < 0.9 for c in chains)
    assert np.all(gelman_rubin(chains) < 1.2)


def test_rejects_invalid_burnin():
    sampler = MetropolisHastings(log_prob=_normal_log_prob, seed=1)
    try:
        sampler.run(np.array([0.0]), n_steps=10, burnin=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
