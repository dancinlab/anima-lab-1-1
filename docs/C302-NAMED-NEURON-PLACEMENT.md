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

## Phase 2 preregistration: runtime dynamics

`C302-NAMED-NEURON-DYNAMICS-1` uses the same five arms and the same pinned
NeuroML source. All protocol values live in the existing config SSOT before the
result is generated:

- seeds 302–306;
- 8-dimensional cell input/output and 16-dimensional hidden state;
- 12 zero-input warmup steps, 6 sensory-stimulus steps, and 12 recovery steps;
- one deterministic unit-norm stimulus vector per seed, delivered only to cells
  whose canonical c302 type contains `sensory`;
- a cloned zero-input sham trajectory from the identical warmed state;
- an exponential spatial kernel whose scale is the actual connectome's median
  observed edge length and is held fixed across every control arm.

The spatial kernel acts before incoming-sum normalization. It changes relative
incoming weights but not the canonical chemical direction or bidirectional gap
junction rule. The imported population and structural edge mask stay locked for
the whole run; no named neuron may be removed, divided, or replaced.

The primary metric is motor response AUC: at each post-warmup step, take the RMS
difference between stimulated and sham per-cell outputs over every canonical
motor neuron, then sum across stimulus and recovery. Secondary readings are
motor peak, sensory AUC, motor/sensory transmission, signed and absolute Phi
delta AUC, tension delta AUC, and sham tension coefficient of variation.

The landing rule is fixed before execution. `actual` must have a higher median
primary metric than every control, must beat each control on at least four of
five paired seeds, and every arm must preserve all 302 cells. If it fails, the
named placement is not promoted as advantageous. Phi and tension remain
descriptive secondary measurements and cannot rescue a failed primary result.

This tests whether the imported structure carries load in the Anima cell
substrate. It does not reproduce c302 membrane biophysics, simulate muscles, or
establish consciousness.

## Phase 2 result

Executed over all 25 pre-registered arm/seed combinations. The run completed
with all 302 named cells preserved in every stimulated and sham trajectory. A
second full execution reproduced every result field exactly after excluding the
timestamp.

| arm | median motor response AUC | actual paired wins |
|---|---:|---:|
| actual | 0.226631 | — |
| connection shuffle | 0.226543 | 1/5 |
| flat | 0.226693 | 2/5 |
| position shuffle | 0.226614 | 3/5 |
| random | 0.226577 | 1/5 |

**Verdict: failed.** Actual/best-control is 0.999725 and no control comparison
reaches the required 4/5 paired wins. The canonical placement therefore has no
measured advantage under this protocol. Secondary Phi, tension, peak-response,
and transmission readings do not override that result.

The run also exposed an instrument limitation that was not used to alter the
verdict. Canonical c302 types contain 111 sensory and 147 motor neurons, with 27
neurons carrying both roles. Because phase 2 stimulated every sensory-labelled
cell and read every motor-labelled cell, those 27 cells contribute direct input
response to the primary motor AUC. The near-identical arm values are therefore
not strong evidence that topology never matters; they show that this registered
readout does not separate topology-mediated transmission from direct dual-role
activation. Any exclusive-motor readout must be a separately registered
experiment, not a rewritten phase 2 score.

Canonical result: `state/c302-named-neuron-dynamics.json`.

## Phase 3 preregistration: exclusive-motor readout

`C302-EXCLUSIVE-MOTOR-DYNAMICS-1` is registered before its first result. It
reuses the pinned source, five control arms, seeds, warmed runtime state,
stimulus, spatial kernel, and landing rule from phase 2. The generic dynamics
selector is configured from the same JSON SSOT; neuron names are not copied
into code or a second list.

The sole experimental change is the readout population. Stimulation still goes
to all 111 neurons carrying the canonical `sensory` role. The primary readout
includes neurons carrying `motor` and excludes every neuron also carrying
`sensory`, yielding 120 exclusive-motor neurons. Therefore no primary-readout
cell receives the external stimulus directly. The primary metric is their
stimulated-versus-sham response AUC.

The result lands only if `actual` has a higher median primary metric than every
control, beats each control in at least four of five paired seeds, and all arms
preserve 302 cells. Phase 2 remains immutable evidence; phase 3 neither replaces
nor rescues its failed verdict. This phase isolates topology-mediated runtime
transmission, but still does not reproduce membrane biophysics or muscles.

Canonical result target: `state/c302-exclusive-motor-dynamics.json`. No result
had been generated when this protocol was registered.

## Run

```bash
python -m c302_placement fetch
python -m c302_placement run
python -m c302_placement dynamics
python -m c302_placement dynamics --experiment-id C302-NAMED-NEURON-DYNAMICS-1
pytest -q tests/test_c302_placement.py
```
