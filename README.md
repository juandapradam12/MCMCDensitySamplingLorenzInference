# MCMC Lab — Bayesian Inference for the Lorenz System

Portfolio case study in computational statistics: Metropolis-Hastings MCMC for (1) sampling a known density and (2) inferring Lorenz-63 parameters from a noisy trajectory.

Originally academic coursework; reworked into a small, reproducible research codebase with clean module boundaries, diagnostics, tests, and CI.

> **Suggested GitHub rename:** `mcmc-lorenz-inference` (current remote name still reflects the original exam repo). Suggested topics: `mcmc`, `bayesian-inference`, `lorenz`, `python`, `scientific-computing`.

## What this shows

| Use case | Question | Output |
| --- | --- | --- |
| Sampler validation | Can MH recover \(N(0,1)\)? | Multi-chain histograms vs analytical density |
| Parameter inference | What \((\sigma, \rho, \beta)\) explain the observed path? | Posterior summaries, traces, predictive fit |

Supporting pieces: Gelman-Rubin \(\hat{R}\), effective sample size (ESS), acceptance rates, a C/`ctypes` multi-chain Normal sampler, and pytest + GitHub Actions CI.

## Project structure

```text
src/mcmc_lab/   sampler, Lorenz model, diagnostics, C bridge, plotting, CLI
c_sampler/      C library + CLI for Normal-target MH (independent chains)
data/           synthetic + legacy observational trajectories
examples/       runnable entry points
tests/          pytest suite
docs/           use-case, architecture, results snapshot
.github/        CI workflow
outputs/        generated figures
```

See [docs/use-case.md](docs/use-case.md), [docs/architecture.md](docs/architecture.md), and [docs/results.md](docs/results.md).

## Setup

```bash
python3 -m pip install -e ".[dev]"
make c-sampler   # optional shared library for the C backend
```

## Run

```bash
make test

python -m mcmc_lab sample-normal
python -m mcmc_lab infer-lorenz
python -m mcmc_lab sample-normal-c 2000 4   # requires make c-sampler
```

Figures land under `outputs/`. Checked-in examples:

<p align="center">
  <img src="docs/figures/normal_sampling.png" width="420" alt="Normal sampling validation" />
  <img src="docs/figures/lorenz_fit.png" width="420" alt="Lorenz trajectory fit" />
</p>

Default-seed numerical summaries are recorded in [docs/results.md](docs/results.md).

## Method (short)

- **Target 1:** \(\log p(x) = -\frac{1}{2}x^2 + const\), random-walk proposals.
- **Target 2:** Lorenz-63 ODE integrated with SciPy; Gaussian likelihood on state residuals; box prior on parameters.
- **Diagnostics:** acceptance rate, multi-chain \(\hat{R}\), and ESS.

Primary inference data: short synthetic Lorenz-63 path with known truth \((10,\ 28,\ 8/3)\) and Gaussian noise. Archival assignment data remains at `data/legacy_observations.dat`.

## Stack

Python 3.10+ · NumPy · SciPy · Matplotlib · pytest · GCC (optional OpenMP) · GitHub Actions

## License

MIT — Juan David Prada Malagon
