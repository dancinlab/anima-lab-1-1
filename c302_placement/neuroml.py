"""NeuroML 2 adapter for named c302 populations and projections."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .model import Connection, Connectome, Neuron, Position


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
    for child in network:
        if _local_name(child.tag) != "population":
            continue
        if neuron_component_contains not in child.attrib.get("component", ""):
            continue
        found = _population_neurons(child)
        neurons.extend(found)
        populations[child.attrib["id"]] = [neuron.neuron_id for neuron in found]

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
    )
    connectome.validate()
    return connectome
