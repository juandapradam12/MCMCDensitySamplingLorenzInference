import numpy as np

from mcmc_lab.lorenz import (
    LorenzParameters,
    integrate_lorenz,
    make_lorenz_log_posterior,
)


def test_classic_lorenz_is_finite_and_chaotic_scale():
    params = LorenzParameters(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    times = np.linspace(0.0, 1.0, 21)
    traj = integrate_lorenz(params, np.array([-4.41, -5.13, -2.93]), times)

    assert traj.shape == (21, 3)
    assert np.isfinite(traj).all()
    assert traj.std() > 1.0


def test_log_posterior_prefers_classic_over_far_params(tmp_path):
    params = LorenzParameters(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    times = np.linspace(0.0, 1.5, 16)
    truth = integrate_lorenz(params, np.array([1.0, 1.0, 1.0]), times)
    rng = np.random.default_rng(0)
    obs = truth + rng.normal(0.0, 0.2, size=truth.shape)

    log_post = make_lorenz_log_posterior(times, obs, sigma_obs=0.5)
    near = log_post(np.array([10.0, 28.0, 8.0 / 3.0]))
    far = log_post(np.array([2.0, 5.0, 1.0]))

    assert np.isfinite(near)
    assert near > far
