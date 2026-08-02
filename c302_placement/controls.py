"""Deterministic placement and topology controls."""

from __future__ import annotations

import random
from dataclasses import replace

from .model import Connectome, Position


def _replace_positions(connectome: Connectome, positions: list[Position]) -> Connectome:
    return replace(
        connectome,
        neurons=tuple(
            replace(neuron, position=position)
            for neuron, position in zip(connectome.neurons, positions, strict=True)
        ),
    )


def shuffle_positions(connectome: Connectome, rng: random.Random) -> Connectome:
    positions = [neuron.position for neuron in connectome.neurons]
    rng.shuffle(positions)
    return _replace_positions(connectome, positions)


def flat_positions(connectome: Connectome) -> Connectome:
    ys = [neuron.position.y for neuron in connectome.neurons]
    low, high = min(ys), max(ys)
    denominator = max(len(ys) - 1, 1)
    positions = [
        Position(0.0, low + (high - low) * index / denominator, 0.0)
        for index in range(len(ys))
    ]
    return _replace_positions(connectome, positions)


def random_positions(connectome: Connectome, rng: random.Random) -> Connectome:
    axes = tuple(
        (min(values), max(values))
        for values in (
            [neuron.position.x for neuron in connectome.neurons],
            [neuron.position.y for neuron in connectome.neurons],
            [neuron.position.z for neuron in connectome.neurons],
        )
    )
    positions = [
        Position(*(rng.uniform(low, high) for low, high in axes))
        for _ in connectome.neurons
    ]
    return _replace_positions(connectome, positions)


def degree_signature(connectome: Connectome) -> dict[str, tuple[int, int, int]]:
    signature = {neuron.neuron_id: [0, 0, 0] for neuron in connectome.neurons}
    for edge in connectome.connections:
        if edge.directed:
            signature[edge.source][0] += 1
            signature[edge.target][1] += 1
        else:
            signature[edge.source][2] += 1
            signature[edge.target][2] += 1
    return {neuron_id: tuple(counts) for neuron_id, counts in signature.items()}


def synapse_degree_signature(
    connectome: Connectome,
) -> dict[tuple[str, str], tuple[int, int, int]]:
    signature = {
        (neuron.neuron_id, mechanism.mechanism_id): [0, 0, 0]
        for neuron in connectome.neurons
        for mechanism in connectome.synapse_mechanisms
    }
    for edge in connectome.connections:
        if edge.directed:
            signature[(edge.source, edge.synapse)][0] += 1
            signature[(edge.target, edge.synapse)][1] += 1
        else:
            signature[(edge.source, edge.synapse)][2] += 1
            signature[(edge.target, edge.synapse)][2] += 1
    return {key: tuple(counts) for key, counts in signature.items()}


def rewire_connections(
    connectome: Connectome, rng: random.Random, swaps_per_edge: float
) -> Connectome:
    edges = list(connectome.connections)
    occupied = {(edge.synapse, edge.source, edge.target) for edge in edges}
    target_swaps = round(len(edges) * swaps_per_edge)
    accepted = 0
    attempts = 0
    while accepted < target_swaps and attempts < max(target_swaps * 20, 1):
        attempts += 1
        left_index, right_index = rng.sample(range(len(edges)), 2)
        left, right = edges[left_index], edges[right_index]
        if left.synapse != right.synapse or left.directed != right.directed:
            continue
        if left.directed:
            new_left = (left.source, right.target)
            new_right = (right.source, left.target)
        else:
            left_nodes = [left.source, left.target]
            right_nodes = [right.source, right.target]
            rng.shuffle(left_nodes)
            rng.shuffle(right_nodes)
            new_left = tuple(sorted((left_nodes[0], right_nodes[1])))
            new_right = tuple(sorted((right_nodes[0], left_nodes[1])))
        old_keys = {
            (left.synapse, left.source, left.target),
            (right.synapse, right.source, right.target),
        }
        new_keys = {(left.synapse, *new_left), (right.synapse, *new_right)}
        if any(source == target for source, target in (new_left, new_right)):
            continue
        if len(new_keys) != 2 or any(key in occupied - old_keys for key in new_keys):
            continue
        occupied.difference_update(old_keys)
        occupied.update(new_keys)
        edges[left_index] = replace(left, source=new_left[0], target=new_left[1])
        edges[right_index] = replace(right, source=new_right[0], target=new_right[1])
        accepted += 1
    if accepted < target_swaps:
        raise RuntimeError(
            f"accepted only {accepted} of {target_swaps} requested edge swaps"
        )
    rewired = replace(connectome, connections=tuple(edges))
    if degree_signature(rewired) != degree_signature(connectome):
        raise AssertionError("degree-preserving rewiring changed a degree")
    if connectome.synapse_mechanisms and (
        synapse_degree_signature(rewired) != synapse_degree_signature(connectome)
    ):
        raise AssertionError("rewiring changed a synapse-specific degree")
    rewired.validate()
    return rewired


def build_variants(
    connectome: Connectome,
    variant_names: tuple[str, ...],
    seed: int,
    swaps_per_edge: float,
) -> dict[str, Connectome]:
    builders = {
        "actual": lambda: connectome,
        "position_shuffle": lambda: shuffle_positions(connectome, random.Random(seed)),
        "connection_shuffle": lambda: rewire_connections(
            connectome, random.Random(seed), swaps_per_edge
        ),
        "flat": lambda: flat_positions(connectome),
        "random": lambda: random_positions(connectome, random.Random(seed)),
    }
    unknown = set(variant_names) - builders.keys()
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    return {name: builders[name]() for name in variant_names}
