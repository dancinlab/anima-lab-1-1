"""Bridge the canonical connectome into the existing Anima runtime engine."""

from __future__ import annotations

import math
from statistics import median
from typing import TYPE_CHECKING

from .model import Connectome

if TYPE_CHECKING:
    from consciousness_engine import ConsciousnessEngine


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
    positions = {neuron.neuron_id: neuron.position for neuron in connectome.neurons}
    for edge in connectome.connections:
        source, target = index[edge.source], index[edge.target]
        weight = edge.weight
        if spatial_kernel == "exponential":
            if distance_scale is None or distance_scale <= 0:
                raise ValueError("exponential spatial kernel requires a positive scale")
            left, right = positions[edge.source], positions[edge.target]
            length = math.dist((left.x, left.y, left.z), (right.x, right.y, right.z))
            weight *= math.exp(-length / distance_scale)
        elif spatial_kernel != "none":
            raise ValueError(f"unsupported spatial kernel: {spatial_kernel}")
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


def bind_connectome(
    engine: ConsciousnessEngine,
    connectome: Connectome,
    normalization: str,
    lock_structure: bool,
    lock_population: bool = False,
    spatial_kernel: str = "none",
    distance_scale: float | None = None,
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
    return engine
