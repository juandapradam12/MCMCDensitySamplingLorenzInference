# MCMC Lab — Bayesian Inference for the Lorenz System

Portfolio case study: Metropolis-Hastings MCMC for (1) sampling a known density and (2) inferring Lorenz-63 parameters from an observational trajectory.

Originally developed as academic coursework; reworked into a small, reproducible research codebase with clean module boundaries, tests, and diagnostics.

## What this shows

| Use case | Question | Output |
| --- | --- | --- |
| Sampler validation | Can MH recover \(N(0,1)\)? | Multi-chain histograms vs analytical density |
| Parameter inference | What \((\sigma, \rho, \beta)\) explain the observed path? | Posterior summaries, traces, predictive fit |

Supporting pieces: Gelman-Rubin \(\hat{R}\), acceptance rates, a C/OpenMP multi-chain Normal sampler, and a Python package used for the Lorenz experiment.

## Project structure

```text
src/mcmc_lab/   sampler, Lorenz model, diagnostics, plotting, CLI
c_sampler/      C + OpenMP Normal-target MH (independent chains in parallel)
data/           observational trajectory (t, x, y, z)
examples/       runnable entry points
tests/          pytest suite
docs/           use-case and architecture notes
outputs/        generated figures
```

See [docs/use-case.md](docs/use-case.md) and [docs/architecture.md](docs/architecture.md).

## Setup

```bash
python3 -m pip install -e ".[dev]"
```

Optional C sampler:

```bash
make c-sampler
./c_sampler/bin/metropolis_normal 2000 4
```

## Run

```bash
# unit tests
make test

# use case 1 — Normal sampling validation
python -m mcmc_lab sample-normal

# use case 2 — Lorenz-63 Bayesian inference
python -m mcmc_lab infer-lorenz
```

Figures are written under `outputs/`. Checked-in examples:

<p align="center">
  <img src="docs/figures/normal_sampling.png" width="420" alt="Normal sampling validation" />
  <img src="docs/figures/lorenz_fit.png" width="420" alt="Lorenz trajectory fit" />
</p>

## Method (short)

- **Target 1:** \(\log p(x) = -\frac{1}{2}x^2 + const\), random-walk proposals.
- **Target 2:** Lorenz-63 ODE integrated with SciPy; Gaussian likelihood on state residuals; box prior on parameters.
- **Diagnostics:** per-chain acceptance rate and multi-chain \(\hat{R}\).

The primary inference dataset is a short synthetic Lorenz-63 trajectory with known truth \((\sigma, \rho, \beta) = (10,\ 28,\ 8/3)\) and Gaussian observation noise. The CLI recovers those parameters with multi-chain \(\hat{R} \approx 1\) and reports posterior credible intervals. The original assignment file is kept as `data/legacy_observations.dat`.

## Stack

Python 3.10+ · NumPy · SciPy · Matplotlib · pytest · GCC/OpenMP (optional)

## License

MIT — Juan David Prada Malagon
