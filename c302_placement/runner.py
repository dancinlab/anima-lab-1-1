"""Pre-registered experiment runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .controls import build_variants, degree_signature
from .metrics import measure
from .neuroml import load_neuroml
from .spec import ExperimentSpec


def run(spec_path: Path, source_path: Path, output_path: Path) -> dict:
    spec = ExperimentSpec.load(spec_path)
    connectome = load_neuroml(source_path, spec.source.neuron_component_contains)
    variants = build_variants(
        connectome, spec.variants, spec.seed, spec.rewiring_swaps_per_edge
    )
    actual_degrees = degree_signature(connectome)
    results = {
        "experiment_id": spec.experiment_id,
        "source": {
            "repository": spec.source.repository,
            "revision": spec.source.revision,
            "model_path": spec.source.model_path,
            "sha256": spec.source.sha256,
        },
        "seed": spec.seed,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "variants": {
            name: {
                **measure(variant),
                "degree_signature_preserved": degree_signature(variant)
                == actual_degrees,
            }
            for name, variant in variants.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
