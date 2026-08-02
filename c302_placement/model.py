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
    conductance_ns: float | None = None


@dataclass(frozen=True, slots=True)
class NamedCell:
    cell_id: str
    component: str
    cell_type: str
    position: Position
    properties: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class GateMechanism:
    gate_id: str
    instances: int
    tau_ms: float | None = None
    midpoint_mv: float | None = None
    scale_mv: float | None = None
    rate: float = 1.0
    calcium_alpha: float | None = None
    calcium_half_mm: float | None = None
    calcium_scale_mm: float | None = None


@dataclass(frozen=True, slots=True)
class IonChannelMechanism:
    channel_id: str
    gates: tuple[GateMechanism, ...]


@dataclass(frozen=True, slots=True)
class ChannelDensity:
    channel_id: str
    conductance_density_ms_cm2: float
    reversal_potential_mv: float
    ion: str


@dataclass(frozen=True, slots=True)
class CellComponent:
    component_id: str
    surface_area_um2: float
    initial_potential_mv: float
    spike_threshold_mv: float
    specific_capacitance_uf_cm2: float
    channel_densities: tuple[ChannelDensity, ...]


@dataclass(frozen=True, slots=True)
class CalciumPool:
    pool_id: str
    resting_concentration_mm: float
    decay_constant_ms: float
    rho_mol_m_a_s: float


@dataclass(frozen=True, slots=True)
class CurrentStimulus:
    stimulus_id: str
    delay_ms: float
    duration_ms: float
    amplitude_pa: float
    target_cell_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NeuromuscularModel:
    source_id: str
    cells: tuple[NamedCell, ...]
    connections: tuple[Connection, ...]
    synapse_mechanisms: tuple[SynapseMechanism, ...]
    cell_components: tuple[CellComponent, ...]
    ion_channels: tuple[IonChannelMechanism, ...]
    calcium_pool: CalciumPool
    stimuli: tuple[CurrentStimulus, ...]
    recommended_timestep_ms: float
    recommended_duration_ms: float

    def validate(self) -> None:
        ids = [cell.cell_id for cell in self.cells]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError(
                "neuromuscular cell identifiers must be nonempty and unique"
            )
        known = set(ids)
        components = {
            component.component_id: component for component in self.cell_components
        }
        channels = {channel.channel_id for channel in self.ion_channels}
        mechanisms = {
            mechanism.mechanism_id: mechanism for mechanism in self.synapse_mechanisms
        }
        if len(components) != len(self.cell_components):
            raise ValueError("cell component identifiers must be unique")
        if any(cell.component not in components for cell in self.cells):
            raise ValueError("a named cell references an unknown component")
        for component in self.cell_components:
            if any(
                density.channel_id not in channels
                for density in component.channel_densities
            ):
                raise ValueError("a channel density references an unknown ion channel")
        for edge in self.connections:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    "a neuromuscular connection references an unknown cell"
                )
            if edge.synapse not in mechanisms:
                raise ValueError(
                    "a neuromuscular connection references an unknown synapse"
                )
        for stimulus in self.stimuli:
            if any(target not in known for target in stimulus.target_cell_ids):
                raise ValueError("a current stimulus references an unknown cell")
        if self.recommended_timestep_ms <= 0 or self.recommended_duration_ms <= 0:
            raise ValueError("recommended simulation timing must be positive")


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
            mechanism.mechanism_id: mechanism for mechanism in self.synapse_mechanisms
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
