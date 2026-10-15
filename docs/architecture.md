# Architecture

The project is split into a Python inference package and a small C sampler used for the distribution-sampling use case.

## Layout

```text
src/mcmc_lab/     Python library (sampler, Lorenz model, diagnostics, plots, CLI)
c_sampler/        OpenMP-capable C implementation of multi-chain Normal sampling
data/             Observational Lorenz trajectory
examples/         Thin wrappers around the CLI entry points
tests/            Unit and smoke tests
docs/             Use-case and architecture notes
outputs/          Generated figures (ignored by git except .gitkeep)
```

## Design choices

### Valid Markov chains

Each chain advances sequentially. Parallelism is applied across independent chains (C/OpenMP), never across proposals inside the same chain. That keeps the Metropolis transition mathematically valid.

### Separation of concerns

| Module | Responsibility |
| --- | --- |
| `metropolis.py` | Generic random-walk Metropolis-Hastings |
| `lorenz.py` | Lorenz-63 ODE, observation loader, log-posterior factory |
| `diagnostics.py` | R-hat, pooled posterior, summary statistics |
| `plotting.py` | Figures for sampler validation and inference |
| `cli.py` | Reproducible experiment entry points |

### Reproducibility

Samplers accept explicit seeds. CLI defaults are fixed so figures and numerical summaries can be regenerated consistently.

### Extensibility

`MetropolisHastings` depends only on a `log_prob(theta)` callable and an optional proposal. New targets (other ODEs, regression models, hierarchical priors) plug in without changing the sampler core.
