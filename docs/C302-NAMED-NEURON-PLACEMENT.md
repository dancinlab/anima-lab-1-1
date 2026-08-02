# C302-NAMED-NEURON-PLACEMENT-1

## Question

Does the named, spatially embedded *C. elegans* connectome provide a measurable
advantage over matched placement and topology controls when used as an Anima
cell substrate?

## Source and SSOT

The canonical input is OpenWorm c302 NeuroML 2. The repository revision, model
path, checksum, random seed, control arms, and rewiring budget live only in
`config/c302_named_neuron_placement.json`. Generated NeuroML is fetched into the
ignored `.cache/c302` directory; it is never copied or manually transcribed.

The adapter preserves neuron names, declared cell type, 3D location, chemical
projection direction, electrical projections, synapse identifier, and weight.
Muscle populations are excluded by the component selector in the SSOT.

`ConsciousnessEngine.configure_topology` is the runtime boundary. The adapter
maps each chemical source into the target cell's incoming coupling and maps gap
junctions in both directions. Incoming weights are normalized by the configured
SSOT rule. A structural mask lets Hebbian plasticity update existing synapses
without silently creating a complete graph.

## Pre-registered arms

1. `actual`: original positions and original connections.
2. `position_shuffle`: positions permuted across fixed named neurons.
3. `connection_shuffle`: degree-preserving edge swaps with positions fixed.
4. `flat`: names and connections fixed on a one-dimensional body axis.
5. `random`: names and connections fixed in the observed 3D bounding box.

The preflight metrics are neuron/connection counts, edge-length distribution,
sensory-to-motor reachability, and sensory-to-motor hop count. They verify the
instrument and controls; they are not evidence of consciousness or Phi.

## Landing rule

This stage lands only if:

- NeuroML parsing recovers 302 named neuron populations;
- all connections resolve to imported neurons;
- every topology control preserves the intended degree signature;
- repeated runs from the SSOT are deterministic except for the recorded time;
- the focused parser, control, and runtime topology QA remains green.

Runtime Phi, tension stability, and stimulated sensory-to-motor transmission
are the next stage. No claim about those outcomes is made by this preflight.

## Run

```bash
python -m c302_placement fetch
python -m c302_placement run
pytest -q tests/test_c302_placement.py
```
