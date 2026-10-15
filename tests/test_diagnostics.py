import numpy as np

from mcmc_lab.diagnostics import effective_sample_size, gelman_rubin, summarize_chain
from mcmc_lab.metropolis import ChainResult


def _fake_chain(mean: float, seed: int) -> ChainResult:
    rng = np.random.default_rng(seed)
    samples = mean + rng.normal(0.0, 1.0, size=(500, 1))
    return ChainResult(
        samples=samples,
        log_prob=np.zeros(500),
        acceptance_rate=0.4,
        n_burnin=100,
    )


def test_gelman_rubin_near_one_for_compatible_chains():
    chains = [_fake_chain(0.0, seed) for seed in (1, 2, 3, 4)]
    rhat = gelman_rubin(chains)
    assert rhat.shape == (1,)
    assert rhat[0] < 1.05


def test_summarize_chain_shapes():
    chain = _fake_chain(1.0, seed=9)
    summary = summarize_chain(chain)
    assert summary.mean.shape == (1,)
    assert summary.q025[0] < summary.mean[0] < summary.q975[0]


def test_ess_near_n_for_independent_draws():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(2000, 1))
    ess = effective_sample_size(draws)
    assert ess.shape == (1,)
    assert ess[0] > 1000


def test_ess_sums_across_chains():
    chains = [_fake_chain(0.0, seed) for seed in (1, 2, 3)]
    ess = effective_sample_size(chains)
    assert ess.shape == (1,)
    assert ess[0] > 500
