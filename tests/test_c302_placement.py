import random
from pathlib import Path

import pytest

from c302_placement.controls import (
    degree_signature,
    flat_positions,
    random_positions,
    rewire_connections,
    shuffle_positions,
)
from c302_placement.metrics import measure
from c302_placement.neuroml import load_neuroml

FIXTURE = Path(__file__).parent / "fixtures" / "c302_minimal.net.nml"


def test_neuroml_adapter_preserves_named_neurons_positions_and_synapse_kinds():
    model = load_neuroml(FIXTURE, "Neuron")

    assert [neuron.neuron_id for neuron in model.neurons] == ["INT", "MOTOR", "SENS"]
    assert [edge.kind for edge in model.connections] == ["chemical", "electrical"]
    assert [edge.weight for edge in model.connections] == [2.0, 3.0]
    assert model.neurons[0].position.x == 1.0


def test_placement_controls_keep_names_and_connections():
    model = load_neuroml(FIXTURE, "Neuron")
    shuffled = shuffle_positions(model, random.Random(302))
    flat = flat_positions(model)
    randomized = random_positions(model, random.Random(302))

    expected_ids = [neuron.neuron_id for neuron in model.neurons]
    assert [neuron.neuron_id for neuron in shuffled.neurons] == expected_ids
    assert shuffled.connections == model.connections
    assert flat.connections == model.connections
    assert randomized.connections == model.connections
    assert all(neuron.position.x == neuron.position.z == 0.0 for neuron in flat.neurons)


def test_rewiring_preserves_directed_and_electrical_degrees(full_connectome):
    rewired = rewire_connections(full_connectome, random.Random(302), 0.25)

    assert degree_signature(rewired) == degree_signature(full_connectome)
    assert rewired.connections != full_connectome.connections


def test_metrics_find_sensory_to_motor_path():
    metrics = measure(load_neuroml(FIXTURE, "Neuron"))

    assert metrics["sensory_motor_reachability"] == 1.0
    assert metrics["median_sensory_motor_hops"] == 2


def test_full_c302_shape(full_connectome):
    assert len(full_connectome.neurons) == 302
    assert len(full_connectome.connections) == 3363
    assert {edge.kind for edge in full_connectome.connections} == {
        "chemical",
        "electrical",
    }


def test_runtime_binding_locks_named_topology():
    torch = pytest.importorskip("torch")
    from c302_placement.runtime import bind_connectome
    from c302_placement.spec import ExperimentSpec
    from consciousness_engine import ConsciousnessEngine

    model = load_neuroml(FIXTURE, "Neuron")
    spec = ExperimentSpec.load(
        Path(__file__).parents[1] / "config" / "c302_named_neuron_placement.json"
    )
    engine = ConsciousnessEngine(
        cell_dim=4,
        hidden_dim=4,
        initial_cells=len(model.neurons),
        max_cells=len(model.neurons),
        phi_ratchet=False,
    )
    bind_connectome(
        engine,
        model,
        spec.runtime.coupling_normalization,
        spec.runtime.lock_structure,
    )

    assert [cell["external_id"] for cell in engine.status()["cells"]] == [
        "INT",
        "MOTOR",
        "SENS",
    ]
    structural_zeros = ~engine._coupling_mask
    engine._hebbian_update(torch.randn(len(model.neurons), engine.cell_dim))
    assert torch.count_nonzero(engine._coupling[structural_zeros]) == 0


def test_full_c302_binds_all_named_runtime_cells(full_connectome):
    torch = pytest.importorskip("torch")
    from c302_placement.runtime import bind_connectome
    from c302_placement.spec import ExperimentSpec
    from consciousness_engine import ConsciousnessEngine

    spec = ExperimentSpec.load(
        Path(__file__).parents[1] / "config" / "c302_named_neuron_placement.json"
    )
    engine = ConsciousnessEngine(
        cell_dim=2,
        hidden_dim=2,
        initial_cells=len(full_connectome.neurons),
        max_cells=len(full_connectome.neurons),
        phi_ratchet=False,
    )
    bind_connectome(
        engine,
        full_connectome,
        spec.runtime.coupling_normalization,
        spec.runtime.lock_structure,
    )

    assert engine.n_cells == 302
    assert torch.count_nonzero(engine._coupling_mask) > 0
    assert engine._coupling.abs().sum(dim=1).max().item() <= 1.0 + 1e-6
    assert {cell["external_id"] for cell in engine.status()["cells"]} == {
        neuron.neuron_id for neuron in full_connectome.neurons
    }
