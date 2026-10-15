"""python -m mcmc_lab <command>"""

from __future__ import annotations

import sys

from mcmc_lab.cli import infer_lorenz_main, sample_normal_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m mcmc_lab {sample-normal|infer-lorenz} [options]")
        return 0 if args else 1

    command, *rest = args
    if command == "sample-normal":
        return sample_normal_main(rest)
    if command == "infer-lorenz":
        return infer_lorenz_main(rest)

    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
