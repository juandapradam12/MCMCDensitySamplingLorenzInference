"""Optional ctypes bridge to the C multi-chain Normal sampler."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class CSamplerUnavailable(RuntimeError):
    """Raised when the shared library cannot be loaded."""


def _default_lib_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "c_sampler" / "bin" / "libmetropolis.so"


class NormalTargetConfig(ctypes.Structure):
    _fields_ = [
        ("mu", ctypes.c_double),
        ("sigma", ctypes.c_double),
        ("proposal_scale", ctypes.c_double),
        ("seed", ctypes.c_uint),
    ]


def load_library(path: str | Path | None = None) -> ctypes.CDLL:
    lib_path = Path(path) if path is not None else _default_lib_path()
    if not lib_path.exists():
        raise CSamplerUnavailable(
            f"Shared library not found at {lib_path}. Run `make c-sampler` first."
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.metropolis_normal_chains.argtypes = [
        ctypes.POINTER(NormalTargetConfig),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.metropolis_normal_chains.restype = ctypes.c_int
    return lib


def sample_normal_c(
    n_samples: int = 2000,
    n_chains: int = 4,
    mu: float = 0.0,
    sigma: float = 1.0,
    proposal_scale: float = 1.0,
    seed: int = 42,
    lib_path: str | Path | None = None,
) -> np.ndarray:
    """Return samples with shape ``(n_chains, n_samples)`` from the C backend."""
    if n_samples < 1 or n_chains < 1:
        raise ValueError("n_samples and n_chains must be >= 1")

    lib = load_library(lib_path)
    config = NormalTargetConfig(
        mu=float(mu),
        sigma=float(sigma),
        proposal_scale=float(proposal_scale),
        seed=ctypes.c_uint(seed),
    )
    out = np.empty((n_chains, n_samples), dtype=np.float64)
    rc = lib.metropolis_normal_chains(
        ctypes.byref(config),
        ctypes.c_size_t(n_chains),
        ctypes.c_size_t(n_samples),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise RuntimeError("C sampler returned a non-zero status")
    return out
