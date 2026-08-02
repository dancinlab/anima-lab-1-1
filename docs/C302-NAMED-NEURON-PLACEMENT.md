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

The canonical result target was registered as
`state/c302-exclusive-motor-dynamics.json`. No result had been generated when
the protocol landed in commit `4fff9db`.

## Phase 3 result

All 25 arm/seed combinations completed with 111 stimulated sensory neurons,
120 exclusive-motor readout neurons, and all 302 named cells preserved. A
second complete run reproduced every field after excluding the timestamp.

| arm | median exclusive-motor response AUC | actual paired wins |
|---|---:|---:|
| actual | 0.000735326 | — |
| connection shuffle | 0.000947304 | 0/5 |
| flat | 0.000843947 | 2/5 |
| position shuffle | 0.000778714 | 1/5 |
| random | 0.000822533 | 1/5 |

**Verdict: failed.** Actual/best-control is 0.776230, and no paired comparison
reaches 4/5 wins. The canonical c302 placement is therefore not advantageous
under this runtime protocol; the degree-preserving connection shuffle has the
highest median topology-mediated response.

The actual-arm median retains only 0.324% of phase 2's all-motor readout, a
99.68% decrease. The populations differ, so this ratio is diagnostic rather
than a new landing criterion, but its scale confirms that the 27 directly
stimulated dual-role neurons dominated phase 2. Phase 3 successfully isolates
a nonzero propagated response and shows that the imported runtime topology
carries signal; it does not show an advantage for the biological arrangement.

Canonical result: `state/c302-exclusive-motor-dynamics.json`.

## Phase 4 preregistration: canonical synapse channels

`C302-SIGNED-SYNAPSE-DYNAMICS-1` is registered before its first result. It
tests whether the failed phase 3 verdict was caused by collapsing every c302
edge into the same positive, instantaneous coupling. The pinned NeuroML source
is the sole authority for each connection's synapse identifier and for each
synapse definition. No neuron, transmitter, receptor, or edge polarity list is
copied into experiment code or configuration.

The runtime adapter must preserve the source-declared neuron-to-neuron
mechanisms as separate channels: excitatory `expTwoSynapse`, inhibitory
`expTwoSynapse`, and bidirectional electrical `gapJunction`. Chemical channel
sign follows the declared reversal potential relative to the source cell's
initial membrane potential; rise and decay constants come from the NeuroML
definition. Electrical coupling is diffusive rather than a second positive
chemical edge. The registered runtime timestep is 1 ms; this is a channel
filter approximation inside the canonical GRU runtime, not a claim to execute
the full NeuroML membrane model.
The implementation follows NeuroML's published `expTwoSynapse` state equations
and peak normalization: <https://docs.neuroml.org/Userdocs/Schemas/Synapses.html>.

All other causal controls remain phase 3's: the pinned 302-neuron source, five
placement/topology arms, seeds 302-306, sensory stimulation, 120 exclusive-motor
readout neurons, spatial kernel, fixed population, and primary response AUC.
Warmup is 20 steps and recovery is 40 steps so the source's 40 ms inhibitory
decay can be observed. The result lands only if `actual` has a higher median
primary metric than every control, beats every control in at least four of five
paired seeds, and every arm preserves 302 cells. A pass supports the complete
package of signed/filter/diffusive semantics; it will not identify which one is
causal without a later ablation. A failure does not falsify c302 biophysics,
because membrane voltage, ion channels, muscles, and environment remain absent.

Canonical result target: `state/c302-signed-synapse-dynamics.json`. No result
had been generated when this protocol was registered.

## Phase 4 result

The protocol was fixed in commit `c2eefe5` before any phase 4 result existed.
All 25 arm/seed combinations then completed with 111 stimulated sensory
neurons, 120 exclusive-motor readout neurons, and all 302 identities preserved.
The generated result records 2,079 excitatory chemical, 200 inhibitory
chemical, and 1,084 electrical connections and their source-declared channel
parameters. A second complete execution reproduced every arm, summary, and
verdict field exactly.

| arm | median exclusive-motor response AUC | actual paired wins |
|---|---:|---:|
| actual | 0.003110961 | — |
| connection shuffle | 0.005236299 | 0/5 |
| flat | 0.007967511 | 0/5 |
| position shuffle | 0.004867838 | 0/5 |
| random | 0.005575627 | 0/5 |

**Verdict: failed.** Actual/best-control is 0.390456, below the required
greater-than-one median ratio, and every paired comparison misses the required
4/5 wins. The signed/filter/diffusive channel package therefore does not rescue
the canonical c302 arrangement in the present homogeneous GRU substrate.

Phase 4 also uses a longer warmup and recovery window, so its absolute AUC and
ratio cannot be interpreted as a controlled ablation against phase 3. The
remaining mechanistic gap is explicit: all 302 neurons still share one GRU cell
model, and the experiment has no membrane voltage, ion-channel diversity,
muscles, body, sensory organ, or environmental feedback. The result rejects
this registered runtime approximation, not the biological connectome.

Canonical result: `state/c302-signed-synapse-dynamics.json`.

## Phase 5 preregistration: conductance cells, muscles, and body loop

`C302-NEUROMUSCULAR-BODY-DYNAMICS-1` is registered before its first result.
It replaces the Anima GRU substrate for this experiment with an explicit
conductance runtime and imports the complete pinned c302 C model: 302 named
neurons, 95 body-wall muscles, neuron-neuron and neuron-muscle projections,
cell morphology, capacitance, initial membrane potential, spike threshold,
channel densities, channel gates, calcium pool, synaptic conductance, the
source pulse generator, and its declared input targets. The included
`cell_C.xml` is fetched from the same immutable upstream revision and verified
by the checksum in the existing JSON SSOT.

The source does **not** declare one membrane model per named neuron. All 302
neurons use `GenericNeuronCell`, while all muscles use `GenericMuscleCell`.
This phase therefore tests source-declared component biophysics, not invented
neuron-level diversity. The runtime follows NeuroML's channel-density current,
fixed-tau sigmoid HH gates, fixed-factor calcium pool, event conductance
synapses, and electrical coupling equations. Its timestep and duration are
read from the network's `recommended_dt_ms` and `recommended_duration_ms`
properties rather than duplicated in the experiment configuration.

c302 does not contain an environmental body solver. The registered extension
is consequently a narrow `damped_segment_chain`: the 95 canonical MDL/MDR/MVL/
MVR identities map to their numeric 24 body segments, dorsal-minus-ventral
muscle activation drives segment curvature, and body strain feeds back only to
the source-declared stimulus input population. Every non-source body parameter
is explicit in the JSON SSOT. This layer is an experimental body approximation,
not an OpenWorm biomechanics claim.

Four paired controls are fixed before execution:

1. `actual_closed_loop`: canonical neural and neuromuscular edges with strain
   feedback.
2. `neural_shuffle_closed_loop`: synapse-specific neuron-neuron degrees are
   preserved while partners are rewired; neuromuscular edges stay canonical.
3. `neuromuscular_shuffle_closed_loop`: neuron-neuron edges stay canonical while
   source and target degrees of neuron-muscle projections are preserved under
   rewiring.
4. `actual_open_loop`: canonical edges and body, but strain-to-sensory feedback
   is disabled.

The source pulse and a zero-pulse sham begin from cloned states with the same
seeded 0.01 mV initial-voltage jitter. The primary metric is sham-subtracted
forward displacement at the source-declared duration. The result lands only if
`actual_closed_loop` has a higher median than every control, wins at least four
of five paired seeds against each, all 397 cells remain finite, and both edge
shuffles preserve their registered degree invariants. Muscle contrast,
curvature, traveling-wave coherence, membrane bounds, and spike/event counts
are secondary diagnostics and cannot rescue a failed primary result.

Canonical result target: `state/c302-neuromuscular-body-dynamics.json`. No
phase 5 result had been generated when this protocol was registered.

## Run

```bash
python -m c302_placement fetch
python -m c302_placement run
python -m c302_placement dynamics
python -m c302_placement dynamics --experiment-id C302-NAMED-NEURON-DYNAMICS-1
python -m c302_placement dynamics --experiment-id C302-SIGNED-SYNAPSE-DYNAMICS-1
pytest -q tests/test_c302_placement.py
```
