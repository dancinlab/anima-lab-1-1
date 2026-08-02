"""Graph and geometry metrics shared by every experimental arm."""

from __future__ import annotations

import math
from collections import Counter, deque
from statistics import fmean, median

from .model import Connectome


def _distances_from(connectome: Connectome, source: str) -> dict[str, int]:
    adjacency: dict[str, list[str]] = {
        neuron.neuron_id: [] for neuron in connectome.neurons
    }
    for edge in connectome.connections:
        adjacency[edge.source].append(edge.target)
        if not edge.directed:
            adjacency[edge.target].append(edge.source)
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for target in adjacency[current]:
            if target not in distances:
                distances[target] = distances[current] + 1
                queue.append(target)
    return distances


def measure(connectome: Connectome) -> dict:
    neurons = {neuron.neuron_id: neuron for neuron in connectome.neurons}
    lengths = []
    for edge in connectome.connections:
        left, right = neurons[edge.source].position, neurons[edge.target].position
        lengths.append(math.dist((left.x, left.y, left.z), (right.x, right.y, right.z)))

    sensory = [
        neuron.neuron_id
        for neuron in connectome.neurons
        if "sensory" in neuron.neuron_type.lower()
    ]
    motor = {
        neuron.neuron_id
        for neuron in connectome.neurons
        if "motor" in neuron.neuron_type.lower()
    }
    path_lengths = []
    reachable_pairs = 0
    for source in sensory:
        distances = _distances_from(connectome, source)
        for target in motor:
            if target in distances:
                reachable_pairs += 1
                path_lengths.append(distances[target])
    total_pairs = len(sensory) * len(motor)
    return {
        "neurons": len(connectome.neurons),
        "connections": len(connectome.connections),
        "connection_kinds": dict(
            sorted(Counter(edge.kind for edge in connectome.connections).items())
        ),
        "neuron_types": dict(
            sorted(Counter(neuron.neuron_type for neuron in connectome.neurons).items())
        ),
        "connection_weight": sum(edge.weight for edge in connectome.connections),
        "mean_connection_length": fmean(lengths) if lengths else 0.0,
        "sensory_motor_reachability": reachable_pairs / total_pairs
        if total_pairs
        else 0.0,
        "median_sensory_motor_hops": median(path_lengths) if path_lengths else None,
    }
