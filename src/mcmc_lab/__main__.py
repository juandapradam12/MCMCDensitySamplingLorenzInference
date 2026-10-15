"""python -m mcmc_lab <command>"""

from __future__ import annotations

import sys

from mcmc_lab.cli import infer_lorenz_main, sample_normal_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: python -m mcmc_lab {sample-normal|infer-lorenz|sample-normal-c} [options]"
        )
        return 0 if args else 1

    command, *rest = args
    if command == "sample-normal":
        return sample_normal_main(rest)
    if command == "infer-lorenz":
        return infer_lorenz_main(rest)
    if command == "sample-normal-c":
        from mcmc_lab.c_backend import sample_normal_c

        n_samples = int(rest[0]) if len(rest) > 0 else 2000
        n_chains = int(rest[1]) if len(rest) > 1 else 4
        samples = sample_normal_c(n_samples=n_samples, n_chains=n_chains)
        print(f"shape={samples.shape} mean={samples.mean():.4f} std={samples.std(ddof=1):.4f}")
        return 0

    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
