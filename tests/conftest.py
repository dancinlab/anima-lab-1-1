"""Put the repo root on the import path for every test in this directory.

Modules under test live at the repo root (bench_v2.py, mitosis.py, trinity.py …),
so tests either do a sys.path dance at the top of each file — which forces
imports below code and provokes an E402 suppression — or it happens once here.
Once here.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def full_connectome():
    from c302_placement.neuroml import load_neuroml
    from c302_placement.spec import ExperimentSpec

    spec = ExperimentSpec.load(ROOT / "config" / "c302_named_neuron_placement.json")
    source = spec.fetch(ROOT / ".cache" / "c302" / "c302_C_Full.net.nml")
    return load_neuroml(source, spec.source.neuron_component_contains)
