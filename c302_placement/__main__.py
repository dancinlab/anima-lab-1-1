"""CLI for fetching and running the canonical experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run
from .spec import ExperimentSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "config" / "c302_named_neuron_placement.json"
DEFAULT_SOURCE = ROOT / ".cache" / "c302" / "c302_C_Full.net.nml"
DEFAULT_OUTPUT = ROOT / "state" / "c302-named-neuron-placement.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "run"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    spec = ExperimentSpec.load(args.spec)
    if args.command == "fetch":
        print(spec.fetch(args.source))
        return
    spec.fetch(args.source)
    results = run(args.spec, args.source, args.output)
    print(
        f"{results['experiment_id']}: {len(results['variants'])} variants -> {args.output}"
    )


if __name__ == "__main__":
    main()
