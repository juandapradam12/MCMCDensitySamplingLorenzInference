# Architecture

The project is split into a Python inference package and a small C sampler used for the distribution-sampling use case.

## Layout

```text
src/mcmc_lab/     Python library (sampler, Lorenz model, diagnostics, C bridge, plots, CLI)
c_sampler/        C library + CLI for Normal-target MH (independent chains)
data/             Synthetic primary trajectory + legacy observational file
examples/         Thin wrappers around the CLI entry points
tests/            Unit and smoke tests
docs/             Use-case, architecture, results, figures
.github/          CI workflow
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
| `diagnostics.py` | R-hat, ESS, pooled posterior, summary statistics |
| `plotting.py` | Figures for sampler validation and inference |
| `c_backend.py` | Optional ctypes bridge to `libmetropolis.so` |
| `cli.py` | Reproducible experiment entry points |

### Reproducibility

Samplers accept explicit seeds. CLI defaults are fixed so figures and numerical summaries can be regenerated consistently.

### Extensibility

`MetropolisHastings` depends only on a `log_prob(theta)` callable and an optional proposal. New targets (other ODEs, regression models, hierarchical priors) plug in without changing the sampler core.
