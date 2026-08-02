import copy
import json
import random
from pathlib import Path

import pytest

from c302_placement.controls import (
    degree_signature,
    flat_positions,
    random_positions,
    rewire_connections,
    shuffle_positions,
    synapse_degree_signature,
)
from c302_placement.metrics import measure
from c302_placement.neuroml import load_neuroml
from c302_placement.runtime import (
    connection_length_scale,
    coupling_matrix,
    synapse_channels,
)
from c302_placement.dynamics import _role_indices

FIXTURE = Path(__file__).parent / "fixtures" / "c302_minimal.net.nml"


def test_neuroml_adapter_preserves_named_neurons_positions_and_synapse_kinds():
    model = load_neuroml(FIXTURE, "Neuron")

    assert [neuron.neuron_id for neuron in model.neurons] == ["INT", "MOTOR", "SENS"]
    assert [edge.kind for edge in model.connections] == ["chemical", "electrical"]
    assert [edge.weight for edge in model.connections] == [2.0, 3.0]
    assert model.neurons[0].position.x == 1.0
    assert model.resting_potential_mv == -45.0
    assert [mechanism.mechanism_id for mechanism in model.synapse_mechanisms] == [
        "exc",
        "gap",
    ]


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
    assert synapse_degree_signature(rewired) == synapse_degree_signature(
        full_connectome
    )
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
    assert full_connectome.resting_potential_mv == -45.0
    mechanisms = {
        mechanism.mechanism_id: mechanism
        for mechanism in full_connectome.synapse_mechanisms
    }
    assert mechanisms["neuron_to_neuron_exc_syn"].reversal_potential_mv == 0.0
    assert mechanisms["neuron_to_neuron_inh_syn"].reversal_potential_mv == -60.0
    assert mechanisms["neuron_to_neuron_inh_syn"].decay_time_ms == 40.0


def test_native_synapse_channels_follow_neuroml_semantics(full_connectome):
    channels = {
        channel.name: channel
        for channel in synapse_channels(
            full_connectome,
            "incoming_sum",
            runtime_timestep_ms=1.0,
        )
    }

    assert set(channels) == {
        "neuron_to_neuron_elec_syn",
        "neuron_to_neuron_exc_syn",
        "neuron_to_neuron_inh_syn",
    }
    assert channels["neuron_to_neuron_exc_syn"].gain == 1.0
    assert channels["neuron_to_neuron_exc_syn"].rise_time_steps == 1.0
    assert channels["neuron_to_neuron_exc_syn"].decay_time_steps == 5.0
    assert channels["neuron_to_neuron_inh_syn"].gain == -1.0
    assert channels["neuron_to_neuron_inh_syn"].decay_time_steps == 40.0
    assert channels["neuron_to_neuron_elec_syn"].mode == "diffusive"
    total = sum(channel.coupling for channel in channels.values())
    assert total.sum(dim=1).max().item() <= 1.0 + 1e-6


def test_diffusive_channel_has_no_uniform_state_drive():
    torch = pytest.importorskip("torch")
    from consciousness_engine import ConsciousnessEngine, TopologyChannel

    engine = ConsciousnessEngine(
        cell_dim=2, hidden_dim=2, initial_cells=2, max_cells=2, phi_ratchet=False
    )
    engine.configure_topology_channels(
        [
            TopologyChannel(
                name="gap",
                coupling=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                mode="diffusive",
            )
        ],
        [
            {"external_id": "left"},
            {"external_id": "right"},
        ],
        lock_population=True,
    )

    uniform = torch.ones(2, 2)
    assert torch.equal(engine._topology_channel_input(uniform), torch.zeros(2, 2))


def test_exp_two_channel_has_normalized_difference_of_exponentials():
    torch = pytest.importorskip("torch")
    from consciousness_engine import ConsciousnessEngine, TopologyChannel

    engine = ConsciousnessEngine(
        cell_dim=1, hidden_dim=1, initial_cells=2, max_cells=2, phi_ratchet=False
    )
    engine.configure_topology_channels(
        [
            TopologyChannel(
                name="exc",
                coupling=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
                rise_time_steps=1.0,
                decay_time_steps=5.0,
            )
        ],
        [{"external_id": "target"}, {"external_id": "source"}],
        lock_population=True,
    )

    impulse = torch.tensor([[0.0], [1.0]])
    zero = torch.zeros(2, 1)
    trace = [engine._topology_channel_input(impulse)[0, 0].item()]
    trace.extend(engine._topology_channel_input(zero)[0, 0].item() for _ in range(20))

    assert trace[0] == 0.0
    assert 0.95 <= max(trace) <= 1.01
    assert trace[-1] < 0.05


def test_role_selection_excludes_dual_role_neurons(full_connectome):
    sensory = _role_indices(full_connectome, ("sensory",), ())
    motor = _role_indices(full_connectome, ("motor",), ())
    exclusive_motor = _role_indices(
        full_connectome, ("motor",), ("sensory",)
    )

    assert len(sensory) == 111
    assert len(motor) == 147
    assert len(exclusive_motor) == 120
    assert set(exclusive_motor).isdisjoint(sensory)


def test_dynamics_protocols_share_one_canonical_ssot():
    from c302_placement.spec import ExperimentSpec

    spec = ExperimentSpec.load(
        Path(__file__).parents[1] / "config" / "c302_named_neuron_placement.json"
    )

    assert spec.dynamics.experiment_id == "C302-EXCLUSIVE-MOTOR-DYNAMICS-1"
    phase_2 = spec.dynamics_for("C302-NAMED-NEURON-DYNAMICS-1")
    phase_3 = spec.dynamics_for("C302-EXCLUSIVE-MOTOR-DYNAMICS-1")
    phase_4 = spec.dynamics_for("C302-SIGNED-SYNAPSE-DYNAMICS-1")
    assert phase_2.readout_exclude_roles == ()
    assert phase_3.readout_exclude_roles == ("sensory",)
    assert phase_2.seeds == phase_3.seeds
    assert phase_4.readout_exclude_roles == ("sensory",)
    assert phase_4.synapse_model == "neuroml_native_channels"
    assert phase_4.seeds == phase_3.seeds


def test_neuromuscular_protocol_is_registered_in_canonical_ssot():
    from c302_placement.spec import ExperimentSpec

    spec = ExperimentSpec.load(
        Path(__file__).parents[1] / "config" / "c302_named_neuron_placement.json"
    )
    phase_5 = spec.biophysics

    assert phase_5.experiment_id == "C302-NEUROMUSCULAR-BODY-DYNAMICS-1"
    assert phase_5.synapse_model == "neuroml_event_conductance"
    assert phase_5.body_model == "damped_segment_chain"
    assert phase_5.body_segments == 24
    assert phase_5.controls == (
        "actual_closed_loop",
        "neural_shuffle_closed_loop",
        "neuromuscular_shuffle_closed_loop",
        "actual_open_loop",
    )
    assert phase_5.primary_metric == "touch_evoked_forward_displacement"
    assert spec.source.include_files[0].model_path == "examples/cell_C.xml"


def test_exclusive_motor_result_matches_registered_population():
    result = json.loads(
        (
            Path(__file__).parents[1]
            / "state"
            / "c302-exclusive-motor-dynamics.json"
        ).read_text(encoding="utf-8")
    )

    assert result["experiment_id"] == "C302-EXCLUSIVE-MOTOR-DYNAMICS-1"
    assert result["protocol"]["readout_include_roles"] == ["motor"]
    assert result["protocol"]["readout_exclude_roles"] == ["sensory"]
    assert {
        row["readout_neurons"]
        for arm in result["arms"].values()
        for row in arm
    } == {120}
    assert {
        row["stimulus_neurons"]
        for arm in result["arms"].values()
        for row in arm
    } == {111}


def test_signed_synapse_result_matches_registered_channels():
    result = json.loads(
        (
            Path(__file__).parents[1]
            / "state"
            / "c302-signed-synapse-dynamics.json"
        ).read_text(encoding="utf-8")
    )

    assert result["experiment_id"] == "C302-SIGNED-SYNAPSE-DYNAMICS-1"
    assert result["protocol"]["synapse_model"] == "neuroml_native_channels"
    assert result["protocol"]["resting_potential_mv"] == -45.0
    assert {
        channel["mechanism_id"]: channel["connection_count"]
        for channel in result["protocol"]["synapse_channels"]
    } == {
        "neuron_to_neuron_elec_syn": 1084,
        "neuron_to_neuron_exc_syn": 2079,
        "neuron_to_neuron_inh_syn": 200,
    }
    assert {
        row["readout_neurons"]
        for arm in result["arms"].values()
        for row in arm
    } == {120}
    assert all(
        row["population_preserved"]
        for arm in result["arms"].values()
        for row in arm
    )
    assert result["verdict"]["landing_passed"] is False


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
        lock_population=spec.runtime.lock_population,
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
        lock_population=spec.runtime.lock_population,
    )

    assert engine.n_cells == 302
    assert torch.count_nonzero(engine._coupling_mask) > 0
    assert engine._coupling.abs().sum(dim=1).max().item() <= 1.0 + 1e-6
    assert {cell["external_id"] for cell in engine.status()["cells"]} == {
        neuron.neuron_id for neuron in full_connectome.neurons
    }


def test_spatial_kernel_changes_weights_when_positions_change(full_connectome):
    scale = connection_length_scale(full_connectome)
    actual = coupling_matrix(full_connectome, "incoming_sum", "exponential", scale)
    shuffled = coupling_matrix(
        shuffle_positions(full_connectome, random.Random(302)),
        "incoming_sum",
        "exponential",
        scale,
    )

    assert not actual.equal(shuffled)
    assert actual.abs().sum(dim=1).max().item() <= 1.0 + 1e-6


def test_named_topology_accepts_cell_specific_input_and_locks_population():
    torch = pytest.importorskip("torch")
    from c302_placement.runtime import bind_connectome
    from consciousness_engine import ConsciousnessEngine

    model = load_neuroml(FIXTURE, "Neuron")
    torch.manual_seed(302)
    engine = ConsciousnessEngine(
        cell_dim=4,
        hidden_dim=4,
        initial_cells=len(model.neurons),
        max_cells=5,
        phi_ratchet=False,
    )
    bind_connectome(
        engine,
        model,
        "incoming_sum",
        lock_structure=True,
        lock_population=True,
    )
    sham = engine.step(cell_inputs=torch.zeros(3, 4))
    stimulus = torch.zeros(3, 4)
    stimulus[2, 0] = 1.0
    driven = engine.step(cell_inputs=stimulus)

    assert driven["cell_outputs"].shape == (3, 4)
    assert not torch.equal(sham["cell_outputs"], driven["cell_outputs"])
    for state in engine.cell_states:
        state.tension_history = [1.0] * engine.split_patience
    assert engine._check_splits() == []
    engine._inter_tension_history[(0, 1)] = [0.0] * engine.merge_patience
    assert engine._check_merges() == []
    assert engine.n_cells == 3


def test_cell_specific_input_rejects_wrong_shape():
    torch = pytest.importorskip("torch")
    from consciousness_engine import ConsciousnessEngine

    engine = ConsciousnessEngine(
        cell_dim=4, hidden_dim=4, initial_cells=3, max_cells=3, phi_ratchet=False
    )
    with pytest.raises(ValueError, match="cell_inputs shape"):
        engine.step(cell_inputs=torch.zeros(2, 4))


def test_zero_cell_inputs_preserve_broadcast_runtime_path():
    torch = pytest.importorskip("torch")
    from consciousness_engine import ConsciousnessEngine

    torch.manual_seed(302)
    broadcast = ConsciousnessEngine(
        cell_dim=4, hidden_dim=4, initial_cells=3, max_cells=3, phi_ratchet=False
    )
    explicit = copy.deepcopy(broadcast)
    drive = torch.tensor([0.1, -0.2, 0.3, -0.4])

    original = broadcast.step(x_input=drive)
    upgraded = explicit.step(x_input=drive, cell_inputs=torch.zeros(3, 4))

    assert torch.equal(original["cell_outputs"], upgraded["cell_outputs"])
    assert torch.equal(original["output"], upgraded["output"])


def test_vectorized_coupling_matches_canonical_sum():
    torch = pytest.importorskip("torch")
    from consciousness_engine import PSI_COUPLING, ConsciousnessEngine

    torch.manual_seed(302)
    engine = ConsciousnessEngine(
        cell_dim=4, hidden_dim=4, initial_cells=3, max_cells=3, phi_ratchet=False
    )
    drive = torch.tensor([0.1, -0.2, 0.3, -0.4])
    engine.step(x_input=drive)
    reference = copy.deepcopy(engine)
    expected = []
    for target, (module, state) in enumerate(
        zip(reference.cell_modules, reference.cell_states, strict=True)
    ):
        cell_input = drive.clone()
        for source, source_state in enumerate(reference.cell_states):
            if source != target:
                cell_input += (
                    PSI_COUPLING
                    * reference._coupling[target, source].item()
                    * source_state.hidden
                )
        output, _ = module(cell_input, state.avg_tension, state.hidden)
        expected.append(output)

    result = engine.step(x_input=drive)

    assert torch.allclose(result["cell_outputs"], torch.stack(expected), atol=1e-7)
