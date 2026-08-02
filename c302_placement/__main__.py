"""CLI for fetching and running the canonical experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dynamics import run_dynamics
from .runner import run
from .spec import ExperimentSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "config" / "c302_named_neuron_placement.json"
DEFAULT_SOURCE = ROOT / ".cache" / "c302" / "c302_C_Full.net.nml"
DEFAULT_OUTPUT = ROOT / "state" / "c302-named-neuron-placement.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "run", "dynamics"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()

    spec = ExperimentSpec.load(args.spec)
    if args.command == "fetch":
        print(spec.fetch(args.source))
        return
    spec.fetch(args.source)
    if args.command == "dynamics":
        dynamics = spec.dynamics_for(args.experiment_id)
        output = args.output or ROOT / dynamics.result_path
        results = run_dynamics(
            args.spec,
            args.source,
            output,
            experiment_id=dynamics.experiment_id,
        )
        print(
            f"{results['experiment_id']}: "
            f"landing_passed={results['verdict']['landing_passed']} -> {output}"
        )
        return
    output = args.output or DEFAULT_OUTPUT
    results = run(args.spec, args.source, output)
    print(
        f"{results['experiment_id']}: {len(results['variants'])} variants -> {output}"
    )


if __name__ == "__main__":
    main()
