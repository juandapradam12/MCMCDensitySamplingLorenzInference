PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.PHONY: install test sample-normal infer-lorenz sample-normal-c c-sampler clean figures all

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

sample-normal:
	$(PYTHON) -m mcmc_lab sample-normal

infer-lorenz:
	$(PYTHON) -m mcmc_lab infer-lorenz

sample-normal-c: c-sampler
	$(PYTHON) -m mcmc_lab sample-normal-c 2000 4

c-sampler:
	$(MAKE) -C c_sampler all

figures: sample-normal infer-lorenz

all: install c-sampler test figures

clean:
	$(MAKE) -C c_sampler clean
	rm -f outputs/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true