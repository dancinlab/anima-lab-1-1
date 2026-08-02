"""Bridge the canonical connectome into the existing Anima runtime engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model import Connectome

if TYPE_CHECKING:
    from consciousness_engine import ConsciousnessEngine


def coupling_matrix(connectome: Connectome, normalization: str):
    """Build ``matrix[target, source]`` without importing torch at module load."""
    import torch

    index = {
        neuron.neuron_id: ordinal for ordinal, neuron in enumerate(connectome.neurons)
    }
    matrix = torch.zeros(len(index), len(index), dtype=torch.float32)
    for edge in connectome.connections:
        source, target = index[edge.source], index[edge.target]
        matrix[target, source] += edge.weight
        if not edge.directed:
            matrix[source, target] += edge.weight
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
        coupling_matrix(connectome, normalization),
        metadata,
        lock_structure=lock_structure,
    )
    return engine
