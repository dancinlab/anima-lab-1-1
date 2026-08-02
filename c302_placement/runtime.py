"""Bridge the canonical connectome into the existing Anima runtime engine."""

from __future__ import annotations

import math
from statistics import median
from typing import TYPE_CHECKING

from .model import Connectome

if TYPE_CHECKING:
    from consciousness_engine import ConsciousnessEngine, TopologyChannel


def connection_length_scale(connectome: Connectome) -> float:
    """Return the median observed edge length for a canonical spatial scale."""
    positions = {neuron.neuron_id: neuron.position for neuron in connectome.neurons}
    lengths = [
        math.dist(
            (
                positions[edge.source].x,
                positions[edge.source].y,
                positions[edge.source].z,
            ),
            (
                positions[edge.target].x,
                positions[edge.target].y,
                positions[edge.target].z,
            ),
        )
        for edge in connectome.connections
    ]
    if not lengths:
        raise ValueError("connectome has no connection length scale")
    scale = float(median(lengths))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("connectome connection length scale must be positive")
    return scale


def _edge_weight(
    connectome: Connectome,
    edge,
    spatial_kernel: str,
    distance_scale: float | None,
) -> float:
    weight = edge.weight
    if spatial_kernel == "exponential":
        if distance_scale is None or distance_scale <= 0:
            raise ValueError("exponential spatial kernel requires a positive scale")
        positions = {neuron.neuron_id: neuron.position for neuron in connectome.neurons}
        left, right = positions[edge.source], positions[edge.target]
        length = math.dist((left.x, left.y, left.z), (right.x, right.y, right.z))
        return weight * math.exp(-length / distance_scale)
    if spatial_kernel != "none":
        raise ValueError(f"unsupported spatial kernel: {spatial_kernel}")
    return weight


def coupling_matrix(
    connectome: Connectome,
    normalization: str,
    spatial_kernel: str = "none",
    distance_scale: float | None = None,
):
    """Build ``matrix[target, source]`` without importing torch at module load."""
    import torch

    index = {
        neuron.neuron_id: ordinal for ordinal, neuron in enumerate(connectome.neurons)
    }
    matrix = torch.zeros(len(index), len(index), dtype=torch.float32)
    for edge in connectome.connections:
        source, target = index[edge.source], index[edge.target]
        weight = _edge_weight(connectome, edge, spatial_kernel, distance_scale)
        matrix[target, source] += weight
        if not edge.directed:
            matrix[source, target] += weight
    matrix.fill_diagonal_(0)
    if normalization == "incoming_sum":
        scale = matrix.abs().sum(dim=1, keepdim=True).clamp(min=1.0)
        matrix = matrix / scale
    else:
        raise ValueError(f"unsupported coupling normalization: {normalization}")
    return matrix


def synapse_channels(
    connectome: Connectome,
    normalization: str,
    runtime_timestep_ms: float,
    spatial_kernel: str = "none",
    distance_scale: float | None = None,
) -> list[TopologyChannel]:
    """Translate source-declared NeuroML mechanisms into runtime channels."""
    import torch

    from consciousness_engine import TopologyChannel

    if runtime_timestep_ms <= 0 or not math.isfinite(runtime_timestep_ms):
        raise ValueError("runtime timestep must be positive")
    if connectome.resting_potential_mv is None:
        raise ValueError("native synapse channels require a resting potential")
    mechanisms = {
        mechanism.mechanism_id: mechanism
        for mechanism in connectome.synapse_mechanisms
    }
    used = {edge.synapse for edge in connectome.connections}
    missing = used - mechanisms.keys()
    if missing:
        raise ValueError(f"missing synapse mechanisms: {sorted(missing)}")

    index = {
        neuron.neuron_id: ordinal for ordinal, neuron in enumerate(connectome.neurons)
    }
    matrices = {
        mechanism_id: torch.zeros(len(index), len(index), dtype=torch.float32)
        for mechanism_id in used
    }
    for edge in connectome.connections:
        source, target = index[edge.source], index[edge.target]
        weight = _edge_weight(connectome, edge, spatial_kernel, distance_scale)
        matrices[edge.synapse][target, source] += weight
        if not edge.directed:
            matrices[edge.synapse][source, target] += weight
    for matrix in matrices.values():
        matrix.fill_diagonal_(0)

    if normalization != "incoming_sum":
        raise ValueError(f"unsupported coupling normalization: {normalization}")
    incoming = sum(matrix.abs() for matrix in matrices.values()).sum(
        dim=1, keepdim=True
    ).clamp(min=1.0)
    channels: list[TopologyChannel] = []
    for mechanism_id in sorted(used):
        mechanism = mechanisms[mechanism_id]
        if mechanism.kind == "gap_junction":
            mode = "diffusive"
            gain = 1.0
            rise_steps = decay_steps = 0.0
        elif mechanism.kind == "exp_two":
            if (
                mechanism.reversal_potential_mv is None
                or mechanism.rise_time_ms is None
                or mechanism.decay_time_ms is None
            ):
                raise ValueError(f"incomplete expTwoSynapse: {mechanism_id}")
            if mechanism.reversal_potential_mv == connectome.resting_potential_mv:
                raise ValueError(f"zero-driving-force synapse: {mechanism_id}")
            mode = "source"
            gain = (
                1.0
                if mechanism.reversal_potential_mv > connectome.resting_potential_mv
                else -1.0
            )
            rise_steps = mechanism.rise_time_ms / runtime_timestep_ms
            decay_steps = mechanism.decay_time_ms / runtime_timestep_ms
        else:
            raise ValueError(f"unsupported synapse mechanism: {mechanism.kind}")
        channels.append(
            TopologyChannel(
                name=mechanism_id,
                coupling=matrices[mechanism_id] / incoming,
                mode=mode,
                gain=gain,
                rise_time_steps=rise_steps,
                decay_time_steps=decay_steps,
            )
        )
    return channels


def bind_connectome(
    engine: ConsciousnessEngine,
    connectome: Connectome,
    normalization: str,
    lock_structure: bool,
    lock_population: bool = False,
    spatial_kernel: str = "none",
    distance_scale: float | None = None,
    synapse_model: str = "static_unsigned",
    runtime_timestep_ms: float = 1.0,
) -> ConsciousnessEngine:
    if engine.n_cells != len(connectome.neurons):
        raise ValueError(
            f"runtime has {engine.n_cells} cells; connectome has {len(connectome.neurons)}"
        )
    metadata = [
        {
            "external_id": neuron.neuron_id,
            "cell_type": neuron.neuron_type,
            "position": (neuron.position.x, neuron.position.y, neuron.position.z),
        }
        for neuron in connectome.neurons
    ]
    if synapse_model == "static_unsigned":
        engine.configure_topology(
            coupling_matrix(
                connectome,
                normalization,
                spatial_kernel=spatial_kernel,
                distance_scale=distance_scale,
            ),
            metadata,
            lock_structure=lock_structure,
            lock_population=lock_population,
        )
    elif synapse_model == "neuroml_native_channels":
        engine.configure_topology_channels(
            synapse_channels(
                connectome,
                normalization,
                runtime_timestep_ms,
                spatial_kernel=spatial_kernel,
                distance_scale=distance_scale,
            ),
            metadata,
            lock_structure=lock_structure,
            lock_population=lock_population,
        )
    else:
        raise ValueError(f"unsupported synapse model: {synapse_model}")
    return engine
