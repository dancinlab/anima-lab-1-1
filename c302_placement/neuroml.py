"""NeuroML 2 adapter for named c302 populations and projections."""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

from .model import (
    CalciumPool,
    CellComponent,
    ChannelDensity,
    Connection,
    Connectome,
    CurrentStimulus,
    GateMechanism,
    IonChannelMechanism,
    NamedCell,
    NeuromuscularModel,
    Neuron,
    Position,
    SynapseMechanism,
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _properties(element: ET.Element) -> dict[str, str]:
    return {
        child.attrib["tag"]: child.attrib.get("value", "")
        for child in element
        if _local_name(child.tag) == "property" and "tag" in child.attrib
    }


def _instance_id(cell_reference: str | None) -> str:
    if not cell_reference:
        return "0"
    parts = [part for part in cell_reference.split("/") if part and part != ".."]
    return parts[1] if len(parts) > 1 and parts[1].isdigit() else "0"


def _quantity(value: str | None, unit: str) -> float | None:
    if value is None:
        return None
    compact = value.strip()
    if not compact.endswith(unit):
        raise ValueError(f"expected {unit} quantity, got {value}")
    return float(compact[: -len(unit)].strip())


def _converted_quantity(
    value: str | None,
    conversions: dict[str, float],
) -> float | None:
    if value is None:
        return None
    compact = value.strip()
    for unit in sorted(conversions, key=len, reverse=True):
        if compact.endswith(unit):
            return float(compact[: -len(unit)].strip()) * conversions[unit]
    raise ValueError(f"unsupported quantity: {value}")


def _synapse_mechanisms(root: ET.Element) -> tuple[SynapseMechanism, ...]:
    mechanisms: list[SynapseMechanism] = []
    for element in root:
        kind = _local_name(element.tag)
        if kind == "expTwoSynapse":
            mechanisms.append(
                SynapseMechanism(
                    mechanism_id=element.attrib["id"],
                    kind="exp_two",
                    reversal_potential_mv=_quantity(element.attrib.get("erev"), "mV"),
                    rise_time_ms=_quantity(element.attrib.get("tauRise"), "ms"),
                    decay_time_ms=_quantity(element.attrib.get("tauDecay"), "ms"),
                    conductance_ns=_converted_quantity(
                        element.attrib.get("gbase"), {"nS": 1.0, "pS": 0.001}
                    ),
                )
            )
        elif kind == "gapJunction":
            mechanisms.append(
                SynapseMechanism(
                    mechanism_id=element.attrib["id"],
                    kind="gap_junction",
                    conductance_ns=_converted_quantity(
                        element.attrib.get("conductance"),
                        {"nS": 1.0, "pS": 0.001},
                    ),
                )
            )
    return tuple(sorted(mechanisms, key=lambda mechanism: mechanism.mechanism_id))


def _resting_potential(root: ET.Element, components: set[str]) -> float | None:
    potentials = {
        value
        for cell in root
        if _local_name(cell.tag) == "cell" and cell.attrib.get("id") in components
        for node in cell.iter()
        if _local_name(node.tag) == "initMembPotential"
        for value in [_quantity(node.attrib.get("value"), "mV")]
        if value is not None
    }
    if len(potentials) > 1:
        raise ValueError("neuron components declare different resting potentials")
    return next(iter(potentials), None)


def _population_neurons(population: ET.Element) -> list[Neuron]:
    population_id = population.attrib["id"]
    component = population.attrib.get("component", "")
    properties = _properties(population)
    instances = [child for child in population if _local_name(child.tag) == "instance"]
    neurons: list[Neuron] = []
    for instance in instances:
        location = next(
            (child for child in instance if _local_name(child.tag) == "location"), None
        )
        if location is None:
            raise ValueError(f"population {population_id} has no location")
        instance_id = instance.attrib.get("id", "0")
        neuron_id = (
            population_id if len(instances) == 1 else f"{population_id}:{instance_id}"
        )
        neurons.append(
            Neuron(
                neuron_id=neuron_id,
                component=component,
                neuron_type=properties.get("type", "unknown"),
                position=Position(
                    x=float(location.attrib["x"]),
                    y=float(location.attrib["y"]),
                    z=float(location.attrib["z"]),
                ),
                properties=properties,
            )
        )
    return neurons


def load_neuroml(path: Path, neuron_component_contains: str) -> Connectome:
    root = ET.parse(path).getroot()
    network = next(
        (node for node in root.iter() if _local_name(node.tag) == "network"), None
    )
    if network is None:
        raise ValueError("NeuroML document contains no network")

    neurons: list[Neuron] = []
    populations: dict[str, list[str]] = {}
    neuron_components: set[str] = set()
    for child in network:
        if _local_name(child.tag) != "population":
            continue
        if neuron_component_contains not in child.attrib.get("component", ""):
            continue
        found = _population_neurons(child)
        neurons.extend(found)
        populations[child.attrib["id"]] = [neuron.neuron_id for neuron in found]
        neuron_components.add(child.attrib.get("component", ""))

    connections: list[Connection] = []
    for projection in network:
        projection_kind = _local_name(projection.tag)
        if projection_kind not in {"projection", "electricalProjection"}:
            continue
        source_population = projection.attrib.get("presynapticPopulation", "")
        target_population = projection.attrib.get("postsynapticPopulation", "")
        if source_population not in populations or target_population not in populations:
            continue
        kind = "chemical" if projection_kind == "projection" else "electrical"
        children = [
            child
            for child in projection
            if "connection" in _local_name(child.tag).lower()
        ]
        for ordinal, child in enumerate(children):
            source_instance = _instance_id(
                child.attrib.get("preCellId") or child.attrib.get("preCell")
            )
            target_instance = _instance_id(
                child.attrib.get("postCellId") or child.attrib.get("postCell")
            )
            source_ids = populations[source_population]
            target_ids = populations[target_population]
            source = (
                source_ids[0]
                if len(source_ids) == 1
                else f"{source_population}:{source_instance}"
            )
            target = (
                target_ids[0]
                if len(target_ids) == 1
                else f"{target_population}:{target_instance}"
            )
            connection_id = f"{projection.attrib.get('id', projection_kind)}:{child.attrib.get('id', ordinal)}"
            connections.append(
                Connection(
                    connection_id=connection_id,
                    source=source,
                    target=target,
                    kind=kind,
                    weight=float(child.attrib.get("weight", "1")),
                    synapse=projection.attrib.get(
                        "synapse", child.attrib.get("synapse", "")
                    ),
                    directed=kind == "chemical",
                )
            )

    connectome = Connectome(
        source_id=network.attrib.get("id", path.stem),
        neurons=tuple(sorted(neurons, key=lambda neuron: neuron.neuron_id)),
        connections=tuple(sorted(connections, key=lambda edge: edge.connection_id)),
        synapse_mechanisms=_synapse_mechanisms(root),
        resting_potential_mv=_resting_potential(root, neuron_components),
    )
    connectome.validate()
    return connectome


def _surface_area_um2(cell: ET.Element) -> float:
    segment = next(
        (node for node in cell.iter() if _local_name(node.tag) == "segment"), None
    )
    if segment is None:
        raise ValueError(f"cell {cell.attrib.get('id')} contains no segment")
    proximal = next(
        (node for node in segment if _local_name(node.tag) == "proximal"), None
    )
    distal = next((node for node in segment if _local_name(node.tag) == "distal"), None)
    if distal is None:
        raise ValueError(f"cell {cell.attrib.get('id')} contains no distal point")
    if proximal is None:
        proximal = distal
    diameter = float(distal.attrib["diameter"])
    length = math.dist(
        tuple(float(proximal.attrib[axis]) for axis in ("x", "y", "z")),
        tuple(float(distal.attrib[axis]) for axis in ("x", "y", "z")),
    )
    return math.pi * diameter * diameter if length == 0 else math.pi * diameter * length


def _cell_components(
    root: ET.Element,
    used_components: set[str],
) -> tuple[CellComponent, ...]:
    components: list[CellComponent] = []
    for cell in root:
        if (
            _local_name(cell.tag) != "cell"
            or cell.attrib.get("id") not in used_components
        ):
            continue
        membrane = next(
            (
                node
                for node in cell.iter()
                if _local_name(node.tag) == "membraneProperties"
            ),
            None,
        )
        if membrane is None:
            raise ValueError(f"cell {cell.attrib['id']} has no membrane properties")
        densities = tuple(
            ChannelDensity(
                channel_id=node.attrib["ionChannel"],
                conductance_density_ms_cm2=float(
                    _converted_quantity(
                        node.attrib.get("condDensity"),
                        {"mS_per_cm2": 1.0, "S_per_cm2": 1000.0},
                    )
                ),
                reversal_potential_mv=float(_quantity(node.attrib.get("erev"), "mV")),
                ion=node.attrib.get("ion", "non_specific"),
            )
            for node in membrane
            if _local_name(node.tag) == "channelDensity"
        )
        components.append(
            CellComponent(
                component_id=cell.attrib["id"],
                surface_area_um2=_surface_area_um2(cell),
                initial_potential_mv=float(
                    _quantity(
                        next(
                            node.attrib.get("value")
                            for node in membrane
                            if _local_name(node.tag) == "initMembPotential"
                        ),
                        "mV",
                    )
                ),
                spike_threshold_mv=float(
                    _quantity(
                        next(
                            node.attrib.get("value")
                            for node in membrane
                            if _local_name(node.tag) == "spikeThresh"
                        ),
                        "mV",
                    )
                ),
                specific_capacitance_uf_cm2=float(
                    _converted_quantity(
                        next(
                            node.attrib.get("value")
                            for node in membrane
                            if _local_name(node.tag) == "specificCapacitance"
                        ),
                        {"uF_per_cm2": 1.0},
                    )
                ),
                channel_densities=densities,
            )
        )
    return tuple(sorted(components, key=lambda component: component.component_id))


def _ion_channels(root: ET.Element) -> tuple[IonChannelMechanism, ...]:
    channels: list[IonChannelMechanism] = []
    for channel in root:
        if _local_name(channel.tag) != "ionChannel":
            continue
        gates: list[GateMechanism] = []
        for gate in channel:
            kind = _local_name(gate.tag)
            if kind == "gateHHtauInf":
                time_course = next(
                    child for child in gate if _local_name(child.tag) == "timeCourse"
                )
                steady_state = next(
                    child for child in gate if _local_name(child.tag) == "steadyState"
                )
                if time_course.attrib.get("type") != "fixedTimeCourse":
                    raise ValueError("only fixedTimeCourse gates are supported")
                if steady_state.attrib.get("type") != "HHSigmoidVariable":
                    raise ValueError("only HHSigmoidVariable gates are supported")
                gates.append(
                    GateMechanism(
                        gate_id=gate.attrib["id"],
                        instances=int(gate.attrib["instances"]),
                        tau_ms=float(_quantity(time_course.attrib.get("tau"), "ms")),
                        midpoint_mv=float(
                            _quantity(steady_state.attrib.get("midpoint"), "mV")
                        ),
                        scale_mv=float(
                            _quantity(steady_state.attrib.get("scale"), "mV")
                        ),
                        rate=float(steady_state.attrib.get("rate", "1")),
                    )
                )
            elif kind == "customHGate":
                gates.append(
                    GateMechanism(
                        gate_id=gate.attrib["id"],
                        instances=int(gate.attrib["instances"]),
                        calcium_alpha=float(gate.attrib["alpha"]),
                        calcium_half_mm=float(
                            _quantity(gate.attrib.get("ca_half"), "mM")
                        ),
                        calcium_scale_mm=float(_quantity(gate.attrib.get("k"), "mM")),
                    )
                )
        channels.append(
            IonChannelMechanism(
                channel_id=channel.attrib["id"],
                gates=tuple(gates),
            )
        )
    return tuple(sorted(channels, key=lambda channel: channel.channel_id))


def _calcium_pool(root: ET.Element) -> CalciumPool:
    node = next(
        child
        for child in root
        if _local_name(child.tag) == "fixedFactorConcentrationModel"
    )
    return CalciumPool(
        pool_id=node.attrib["id"],
        resting_concentration_mm=float(
            _converted_quantity(
                node.attrib.get("restingConc"),
                {"mM": 1.0, "mol_per_cm3": 1_000_000.0},
            )
        ),
        decay_constant_ms=float(_quantity(node.attrib.get("decayConstant"), "ms")),
        rho_mol_m_a_s=float(_quantity(node.attrib.get("rho"), "mol_per_m_per_A_per_s")),
    )


def load_neuromuscular_neuroml(
    network_path: Path,
    channel_path: Path,
) -> NeuromuscularModel:
    root = ET.parse(network_path).getroot()
    channel_root = ET.parse(channel_path).getroot()
    network = next(
        (node for node in root.iter() if _local_name(node.tag) == "network"),
        None,
    )
    if network is None:
        raise ValueError("NeuroML document contains no network")

    populations: dict[str, list[str]] = {}
    cells: list[NamedCell] = []
    for population in network:
        if _local_name(population.tag) != "population":
            continue
        found = _population_neurons(population)
        population_cells = [
            NamedCell(
                cell_id=neuron.neuron_id,
                component=neuron.component,
                cell_type=(
                    "muscle" if "Muscle" in neuron.component else neuron.neuron_type
                ),
                position=neuron.position,
                properties=neuron.properties,
            )
            for neuron in found
        ]
        cells.extend(population_cells)
        populations[population.attrib["id"]] = [
            cell.cell_id for cell in population_cells
        ]

    connections: list[Connection] = []
    for projection in network:
        projection_kind = _local_name(projection.tag)
        if projection_kind not in {"projection", "electricalProjection"}:
            continue
        source_population = projection.attrib.get("presynapticPopulation", "")
        target_population = projection.attrib.get("postsynapticPopulation", "")
        if source_population not in populations or target_population not in populations:
            continue
        kind = "chemical" if projection_kind == "projection" else "electrical"
        for ordinal, child in enumerate(
            node for node in projection if "connection" in _local_name(node.tag).lower()
        ):
            source_instance = _instance_id(
                child.attrib.get("preCellId") or child.attrib.get("preCell")
            )
            target_instance = _instance_id(
                child.attrib.get("postCellId") or child.attrib.get("postCell")
            )
            source_ids = populations[source_population]
            target_ids = populations[target_population]
            connections.append(
                Connection(
                    connection_id=(
                        f"{projection.attrib.get('id', projection_kind)}:"
                        f"{child.attrib.get('id', ordinal)}"
                    ),
                    source=(
                        source_ids[0]
                        if len(source_ids) == 1
                        else f"{source_population}:{source_instance}"
                    ),
                    target=(
                        target_ids[0]
                        if len(target_ids) == 1
                        else f"{target_population}:{target_instance}"
                    ),
                    kind=kind,
                    weight=float(child.attrib.get("weight", "1")),
                    synapse=projection.attrib.get(
                        "synapse", child.attrib.get("synapse", "")
                    ),
                    directed=kind == "chemical",
                )
            )

    pulse_generators = {
        node.attrib["id"]: node
        for node in root
        if _local_name(node.tag) == "pulseGenerator"
    }
    stimulus_targets: dict[str, list[str]] = {}
    for input_list in network:
        if _local_name(input_list.tag) != "inputList":
            continue
        component = input_list.attrib["component"]
        population = input_list.attrib["population"]
        stimulus_targets.setdefault(component, []).extend(populations[population])
    stimuli = tuple(
        CurrentStimulus(
            stimulus_id=stimulus_id,
            delay_ms=float(_quantity(node.attrib.get("delay"), "ms")),
            duration_ms=float(_quantity(node.attrib.get("duration"), "ms")),
            amplitude_pa=float(_quantity(node.attrib.get("amplitude"), "pA")),
            target_cell_ids=tuple(sorted(stimulus_targets.get(stimulus_id, []))),
        )
        for stimulus_id, node in sorted(pulse_generators.items())
        if stimulus_targets.get(stimulus_id)
    )
    properties = _properties(network)
    model = NeuromuscularModel(
        source_id=network.attrib.get("id", network_path.stem),
        cells=tuple(
            sorted(cells, key=lambda cell: (cell.cell_type == "muscle", cell.cell_id))
        ),
        connections=tuple(sorted(connections, key=lambda edge: edge.connection_id)),
        synapse_mechanisms=_synapse_mechanisms(root),
        cell_components=_cell_components(root, {cell.component for cell in cells}),
        ion_channels=_ion_channels(channel_root),
        calcium_pool=_calcium_pool(root),
        stimuli=stimuli,
        recommended_timestep_ms=float(properties["recommended_dt_ms"]),
        recommended_duration_ms=float(properties["recommended_duration_ms"]),
    )
    model.validate()
    return model
