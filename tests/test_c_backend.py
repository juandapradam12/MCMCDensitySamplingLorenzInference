import pytest

from mcmc_lab.c_backend import CSamplerUnavailable, sample_normal_c


def test_c_backend_samples_unit_normal_mean():
    try:
        samples = sample_normal_c(n_samples=1500, n_chains=3, seed=11)
    except CSamplerUnavailable:
        pytest.skip("C shared library not built")

    assert samples.shape == (3, 1500)
    burnin = 400
    pooled = samples[:, burnin:].ravel()
    assert abs(pooled.mean()) < 0.2
    assert 0.7 < pooled.std(ddof=1) < 1.3
