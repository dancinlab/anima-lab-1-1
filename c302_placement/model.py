"""Runtime-neutral canonical connectome types.

NeuroML remains the source format. These types are the narrow adapter contract
used by the experiment and, later, by Anima runtime engines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Position:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Neuron:
    neuron_id: str
    component: str
    neuron_type: str
    position: Position
    properties: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Connection:
    connection_id: str
    source: str
    target: str
    kind: str
    weight: float
    synapse: str
    directed: bool


@dataclass(frozen=True, slots=True)
class SynapseMechanism:
    mechanism_id: str
    kind: str
    reversal_potential_mv: float | None = None
    rise_time_ms: float | None = None
    decay_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class Connectome:
    source_id: str
    neurons: tuple[Neuron, ...]
    connections: tuple[Connection, ...]
    synapse_mechanisms: tuple[SynapseMechanism, ...] = ()
    resting_potential_mv: float | None = None

    def validate(self) -> None:
        ids = [neuron.neuron_id for neuron in self.neurons]
        if not ids:
            raise ValueError("connectome contains no neurons")
        if len(ids) != len(set(ids)):
            raise ValueError("neuron identifiers must be unique")
        known = set(ids)
        mechanisms = {
            mechanism.mechanism_id: mechanism
            for mechanism in self.synapse_mechanisms
        }
        if len(mechanisms) != len(self.synapse_mechanisms):
            raise ValueError("synapse mechanism identifiers must be unique")
        connection_ids: set[str] = set()
        for edge in self.connections:
            if edge.connection_id in connection_ids:
                raise ValueError(f"duplicate connection id: {edge.connection_id}")
            connection_ids.add(edge.connection_id)
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"connection {edge.connection_id} references an unknown neuron"
                )
            if edge.kind not in {"chemical", "electrical"}:
                raise ValueError(f"unsupported connection kind: {edge.kind}")
            if edge.weight < 0:
                raise ValueError(f"negative connection weight: {edge.connection_id}")
            if mechanisms and edge.synapse not in mechanisms:
                raise ValueError(
                    f"connection {edge.connection_id} references unknown synapse "
                    f"mechanism {edge.synapse}"
                )

    def to_dict(self) -> dict:
        self.validate()
        return {
            "source_id": self.source_id,
            "neurons": [asdict(neuron) for neuron in self.neurons],
            "connections": [asdict(edge) for edge in self.connections],
            "synapse_mechanisms": [
                asdict(mechanism) for mechanism in self.synapse_mechanisms
            ],
            "resting_potential_mv": self.resting_potential_mv,
        }
