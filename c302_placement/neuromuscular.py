"""Source-driven c302 conductance, muscle, and reduced-body runtime."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

from .controls import rewire_connections
from .model import Connectome, NeuromuscularModel, Neuron
from .neuroml import load_neuromuscular_neuroml
from .spec import BiophysicsSpec, ExperimentSpec

_MUSCLE_ID = re.compile(r"^M([DV])([LR])(\d{2})$")


def _as_connectome(model: NeuromuscularModel, connections) -> Connectome:
    return Connectome(
        source_id=model.source_id,
        neurons=tuple(
            Neuron(
                neuron_id=cell.cell_id,
                component=cell.component,
                neuron_type=cell.cell_type,
                position=cell.position,
                properties=cell.properties,
            )
            for cell in model.cells
        ),
        connections=tuple(connections),
        synapse_mechanisms=model.synapse_mechanisms,
    )


def build_biophysical_controls(
    model: NeuromuscularModel,
    names: tuple[str, ...],
    seed: int,
    swaps_per_edge: float,
) -> dict[str, tuple[NeuromuscularModel, bool]]:
    cell_types = {cell.cell_id: cell.cell_type for cell in model.cells}
    neural = tuple(
        edge
        for edge in model.connections
        if cell_types[edge.source] != "muscle" and cell_types[edge.target] != "muscle"
    )
    neuromuscular = tuple(
        edge
        for edge in model.connections
        if cell_types[edge.source] != "muscle" and cell_types[edge.target] == "muscle"
    )
    if len(neural) + len(neuromuscular) != len(model.connections):
        raise ValueError(
            "the c302 runtime supports neural and neuron-to-muscle edges only"
        )
    neural_shuffled = rewire_connections(
        _as_connectome(model, neural), random.Random(seed), swaps_per_edge
    ).connections
    neuromuscular_shuffled = rewire_connections(
        _as_connectome(model, neuromuscular), random.Random(seed), swaps_per_edge
    ).connections
    builders = {
        "actual_closed_loop": (model, True),
        "neural_shuffle_closed_loop": (
            replace(model, connections=tuple(neural_shuffled) + neuromuscular),
            True,
        ),
        "neuromuscular_shuffle_closed_loop": (
            replace(model, connections=neural + tuple(neuromuscular_shuffled)),
            True,
        ),
        "actual_open_loop": (model, False),
    }
    unknown = set(names) - builders.keys()
    if unknown:
        raise ValueError(f"unknown biophysical controls: {sorted(unknown)}")
    return {name: builders[name] for name in names}


class ConductanceBodyRuntime:
    """Vectorized single-compartment NeuroML subset plus a reduced body chain."""

    def __init__(
        self,
        model: NeuromuscularModel,
        protocol: BiophysicsSpec,
        seed: int,
        feedback_enabled: bool,
    ) -> None:
        import numpy as np

        self.np = np
        self.model = model
        self.protocol = protocol
        self.dt = model.recommended_timestep_ms
        self.feedback_enabled = feedback_enabled
        self.index = {cell.cell_id: index for index, cell in enumerate(model.cells)}
        self.cell_ids = [cell.cell_id for cell in model.cells]
        self.is_muscle = np.array(
            [cell.cell_type == "muscle" for cell in model.cells], dtype=bool
        )
        components = {
            component.component_id: component for component in model.cell_components
        }
        self.components = [components[cell.component] for cell in model.cells]
        rng = np.random.default_rng(seed)
        self.voltage = np.array(
            [component.initial_potential_mv for component in self.components],
            dtype=np.float64,
        )
        self.voltage += rng.normal(
            0.0, protocol.initial_voltage_jitter_mv, size=len(self.voltage)
        )
        self.previous_voltage = self.voltage.copy()
        self.threshold = np.array(
            [component.spike_threshold_mv for component in self.components]
        )
        self.capacitance_pf = np.array(
            [
                component.specific_capacitance_uf_cm2
                * component.surface_area_um2
                * 0.01
                for component in self.components
            ]
        )
        self.capacitance_density = np.array(
            [component.specific_capacitance_uf_cm2 for component in self.components]
        )
        self.area_um2 = np.array(
            [component.surface_area_um2 for component in self.components]
        )
        channel_specs = {channel.channel_id: channel for channel in model.ion_channels}
        channel_ids = sorted(channel_specs)
        self.channel_ids = channel_ids
        self.channel_specs = channel_specs
        self.gbar = {
            channel_id: np.array(
                [
                    next(
                        (
                            density.conductance_density_ms_cm2
                            for density in component.channel_densities
                            if density.channel_id == channel_id
                        ),
                        0.0,
                    )
                    for component in self.components
                ]
            )
            for channel_id in channel_ids
        }
        self.erev = {
            channel_id: np.array(
                [
                    next(
                        (
                            density.reversal_potential_mv
                            for density in component.channel_densities
                            if density.channel_id == channel_id
                        ),
                        0.0,
                    )
                    for component in self.components
                ]
            )
            for channel_id in channel_ids
        }
        self.channel_ions = {
            channel_id: next(
                (
                    density.ion
                    for component in model.cell_components
                    for density in component.channel_densities
                    if density.channel_id == channel_id
                ),
                "non_specific",
            )
            for channel_id in channel_ids
        }
        self.calcium = np.full(
            len(model.cells), model.calcium_pool.resting_concentration_mm
        )
        self.gates: dict[tuple[str, str], object] = {}
        for channel_id, channel in channel_specs.items():
            for gate in channel.gates:
                if gate.tau_ms is not None:
                    self.gates[(channel_id, gate.gate_id)] = self._gate_inf(
                        self.voltage, gate
                    )

        mechanisms = {
            mechanism.mechanism_id: mechanism for mechanism in model.synapse_mechanisms
        }
        chemical = [edge for edge in model.connections if edge.kind == "chemical"]
        electrical = [edge for edge in model.connections if edge.kind == "electrical"]
        self.chemical_groups = []
        for mechanism_id in sorted({edge.synapse for edge in chemical}):
            mechanism = mechanisms[mechanism_id]
            if (
                mechanism.conductance_ns is None
                or mechanism.reversal_potential_mv is None
                or mechanism.rise_time_ms is None
                or mechanism.decay_time_ms is None
            ):
                raise ValueError(f"incomplete chemical mechanism: {mechanism_id}")
            edges = [edge for edge in chemical if edge.synapse == mechanism_id]
            tau_rise = mechanism.rise_time_ms
            tau_decay = mechanism.decay_time_ms
            peak_time = math.log(tau_decay / tau_rise) * (
                tau_rise * tau_decay / (tau_decay - tau_rise)
            )
            factor = 1.0 / (
                math.exp(-peak_time / tau_decay) - math.exp(-peak_time / tau_rise)
            )
            self.chemical_groups.append(
                {
                    "id": mechanism_id,
                    "source": np.array([self.index[edge.source] for edge in edges]),
                    "target": np.array([self.index[edge.target] for edge in edges]),
                    "weight": np.array([edge.weight for edge in edges]),
                    "gbase": mechanism.conductance_ns,
                    "erev": mechanism.reversal_potential_mv,
                    "rise_alpha": math.exp(-self.dt / tau_rise),
                    "decay_alpha": math.exp(-self.dt / tau_decay),
                    "factor": factor,
                    "rise": np.zeros(len(edges)),
                    "decay": np.zeros(len(edges)),
                }
            )
        self.electrical_groups = []
        for mechanism_id in sorted({edge.synapse for edge in electrical}):
            mechanism = mechanisms[mechanism_id]
            if mechanism.conductance_ns is None:
                raise ValueError(f"incomplete electrical mechanism: {mechanism_id}")
            edges = [edge for edge in electrical if edge.synapse == mechanism_id]
            self.electrical_groups.append(
                {
                    "source": np.array([self.index[edge.source] for edge in edges]),
                    "target": np.array([self.index[edge.target] for edge in edges]),
                    "conductance": mechanism.conductance_ns
                    * np.array([edge.weight for edge in edges]),
                }
            )

        stimulus = next(
            item
            for item in model.stimuli
            if item.stimulus_id == protocol.stimulus_component_id
        )
        self.stimulus = stimulus
        self.stimulus_targets = np.array(
            [self.index[cell_id] for cell_id in stimulus.target_cell_ids]
        )
        self.target_segments = np.array(
            [
                self._nearest_segment(model.cells[index].position.y)
                for index in self.stimulus_targets
            ]
        )
        self.target_sides = np.array(
            [
                -1.0 if model.cells[index].cell_id.endswith("L") else 1.0
                for index in self.stimulus_targets
            ]
        )
        self.muscle_activation = np.zeros(len(model.cells))
        self.muscle_segments: list[tuple[list[int], list[int]]] = []
        for segment in range(1, protocol.body_segments + 1):
            dorsal: list[int] = []
            ventral: list[int] = []
            for index, cell_id in enumerate(self.cell_ids):
                match = _MUSCLE_ID.match(cell_id)
                if match and int(match.group(3)) == segment:
                    (dorsal if match.group(1) == "D" else ventral).append(index)
            if not dorsal or not ventral:
                raise ValueError(
                    f"body segment {segment} lacks dorsal or ventral muscle"
                )
            self.muscle_segments.append((dorsal, ventral))
        self.curvature = np.zeros(protocol.body_segments)
        self.curvature_velocity = np.zeros(protocol.body_segments)
        self.forward_displacement = 0.0
        self.event_count = 0
        self.max_abs_voltage = float(np.abs(self.voltage).max())
        self.curvature_samples: list[object] = []
        self.velocity_samples: list[object] = []

    def _nearest_segment(self, y: float) -> int:
        muscle_positions = []
        for segment in range(1, self.protocol.body_segments + 1):
            values = [
                cell.position.y
                for cell in self.model.cells
                if (match := _MUSCLE_ID.match(cell.cell_id))
                and int(match.group(3)) == segment
            ]
            muscle_positions.append(sum(values) / len(values))
        return min(
            range(self.protocol.body_segments),
            key=lambda index: abs(muscle_positions[index] - y),
        )

    def _gate_inf(self, voltage, gate):
        np = self.np
        exponent = np.clip(-(voltage - gate.midpoint_mv) / gate.scale_mv, -700.0, 700.0)
        return gate.rate / (1.0 + np.exp(exponent))

    def _channel_open_fraction(self, channel_id: str):
        np = self.np
        fraction = np.ones(len(self.voltage))
        for gate in self.channel_specs[channel_id].gates:
            if gate.tau_ms is not None:
                fraction *= self.gates[(channel_id, gate.gate_id)] ** gate.instances
            else:
                exponent = np.clip(
                    (gate.calcium_half_mm - self.calcium) / gate.calcium_scale_mm,
                    -700.0,
                    700.0,
                )
                q = 1.0 / (1.0 + np.exp(exponent))
                fcond = 1.0 + (q - 1.0) * gate.calcium_alpha
                fraction *= fcond**gate.instances
        return fraction

    def _synaptic_current(self):
        np = self.np
        current = np.zeros(len(self.voltage))
        for group in self.chemical_groups:
            conductance = (
                group["gbase"] * group["weight"] * (group["decay"] - group["rise"])
            )
            edge_current = conductance * (group["erev"] - self.voltage[group["target"]])
            current += np.bincount(
                group["target"], edge_current, minlength=len(current)
            )
        for group in self.electrical_groups:
            edge_current = group["conductance"] * (
                self.voltage[group["source"]] - self.voltage[group["target"]]
            )
            current += np.bincount(
                group["target"], edge_current, minlength=len(current)
            )
            current -= np.bincount(
                group["source"], edge_current, minlength=len(current)
            )
        return current

    def _body_feedback_current(self):
        np = self.np
        current = np.zeros(len(self.voltage))
        if self.feedback_enabled:
            current[self.stimulus_targets] = (
                self.protocol.proprioceptive_feedback_gain_pa
                * self.target_sides
                * self.curvature[self.target_segments]
            )
        return current

    def _update_body(self, sample: bool) -> None:
        np = self.np
        target_activation = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    -(self.voltage - self.threshold)
                    / self.protocol.muscle_activation_slope_mv,
                    -700.0,
                    700.0,
                )
            )
        )
        alpha = 1.0 - math.exp(
            -self.dt / self.protocol.muscle_activation_time_constant_ms
        )
        self.muscle_activation[self.is_muscle] += alpha * (
            target_activation[self.is_muscle] - self.muscle_activation[self.is_muscle]
        )
        torque = np.array(
            [
                self.muscle_activation[dorsal].mean()
                - self.muscle_activation[ventral].mean()
                for dorsal, ventral in self.muscle_segments
            ]
        )
        laplacian = np.zeros_like(self.curvature)
        laplacian[1:-1] = (
            self.curvature[:-2] - 2.0 * self.curvature[1:-1] + self.curvature[2:]
        )
        acceleration = (
            self.protocol.body_torque_gain * torque
            - self.protocol.body_stiffness * self.curvature
            + 0.5 * self.protocol.body_stiffness * laplacian
            - self.protocol.body_damping * self.curvature_velocity
        )
        self.curvature_velocity += self.dt * acceleration
        self.curvature += self.dt * self.curvature_velocity
        spatial_gradient = np.gradient(self.curvature)
        forward_speed = -float(np.mean(self.curvature_velocity * spatial_gradient))
        self.forward_displacement += self.dt * forward_speed
        if sample:
            self.curvature_samples.append(self.curvature.copy())
            self.velocity_samples.append(self.curvature_velocity.copy())

    def step(self, time_ms: float, stimulus_enabled: bool, sample: bool) -> None:
        np = self.np
        ion_drive = np.zeros(len(self.voltage))
        calcium_drive = np.zeros(len(self.voltage))
        for channel_id in self.channel_ids:
            current_density = (
                self.gbar[channel_id]
                * self._channel_open_fraction(channel_id)
                * (self.erev[channel_id] - self.voltage)
            )
            ion_drive += current_density
            if self.channel_ions[channel_id] == "ca":
                calcium_drive += current_density
        point_current = self._synaptic_current() + self._body_feedback_current()
        if (
            stimulus_enabled
            and self.stimulus.delay_ms
            <= time_ms
            < self.stimulus.delay_ms + self.stimulus.duration_ms
        ):
            point_current[self.stimulus_targets] += self.stimulus.amplitude_pa
        self.previous_voltage[:] = self.voltage
        self.voltage += self.dt * (
            ion_drive / self.capacitance_density + point_current / self.capacitance_pf
        )
        if not np.isfinite(self.voltage).all():
            raise FloatingPointError("non-finite membrane potential")
        self.max_abs_voltage = max(
            self.max_abs_voltage, float(np.abs(self.voltage).max())
        )
        for channel_id, channel in self.channel_specs.items():
            for gate in channel.gates:
                if gate.tau_ms is None:
                    continue
                state = self.gates[(channel_id, gate.gate_id)]
                state += (1.0 - math.exp(-self.dt / gate.tau_ms)) * (
                    self._gate_inf(self.voltage, gate) - state
                )
        pool = self.model.calcium_pool
        self.calcium += self.dt * (
            calcium_drive * pool.rho_mol_m_a_s * 1e-5
            - (self.calcium - pool.resting_concentration_mm) / pool.decay_constant_ms
        )
        self.calcium.clip(min=0.0, out=self.calcium)
        crossed = (self.previous_voltage < self.threshold) & (
            self.voltage >= self.threshold
        )
        self.event_count += int(crossed.sum())
        for group in self.chemical_groups:
            group["rise"] *= group["rise_alpha"]
            group["decay"] *= group["decay_alpha"]
            events = crossed[group["source"]]
            group["rise"][events] += group["factor"]
            group["decay"][events] += group["factor"]
        self._update_body(sample)

    def run(self, stimulus_enabled: bool) -> dict:
        np = self.np
        steps = round(self.model.recommended_duration_ms / self.dt)
        sample_stride = max(round(1.0 / self.dt), 1)
        for step in range(steps):
            self.step(
                step * self.dt,
                stimulus_enabled=stimulus_enabled,
                sample=step % sample_stride == 0,
            )
        curvature = np.stack(self.curvature_samples)
        velocity = np.stack(self.velocity_samples)
        gradient = np.gradient(curvature, axis=1)
        left = velocity.reshape(-1)
        right = (-gradient).reshape(-1)
        coherence = (
            float(np.corrcoef(left, right)[0, 1])
            if left.std() > 0 and right.std() > 0
            else 0.0
        )
        return {
            "forward_displacement": self.forward_displacement,
            "curvature_rms": float(np.sqrt(np.mean(curvature * curvature))),
            "traveling_wave_coherence": coherence,
            "muscle_activation_contrast": float(
                np.std(self.muscle_activation[self.is_muscle])
            ),
            "event_count": self.event_count,
            "max_abs_membrane_potential_mv": self.max_abs_voltage,
            "finite": bool(
                np.isfinite(self.voltage).all()
                and np.isfinite(self.curvature).all()
                and np.isfinite(self.calcium).all()
            ),
        }


def _run_pair(
    model: NeuromuscularModel,
    protocol: BiophysicsSpec,
    seed: int,
    feedback_enabled: bool,
) -> dict:
    stimulated = ConductanceBodyRuntime(model, protocol, seed, feedback_enabled).run(
        stimulus_enabled=True
    )
    sham = ConductanceBodyRuntime(model, protocol, seed, feedback_enabled).run(
        stimulus_enabled=False
    )
    return {
        "seed": seed,
        "touch_evoked_forward_displacement": (
            stimulated["forward_displacement"] - sham["forward_displacement"]
        ),
        "touch_evoked_curvature_rms": (
            stimulated["curvature_rms"] - sham["curvature_rms"]
        ),
        "stimulated_forward_displacement": stimulated["forward_displacement"],
        "sham_forward_displacement": sham["forward_displacement"],
        "stimulated_curvature_rms": stimulated["curvature_rms"],
        "stimulated_traveling_wave_coherence": stimulated["traveling_wave_coherence"],
        "stimulated_muscle_activation_contrast": stimulated[
            "muscle_activation_contrast"
        ],
        "stimulated_event_count": stimulated["event_count"],
        "sham_event_count": sham["event_count"],
        "max_abs_membrane_potential_mv": max(
            stimulated["max_abs_membrane_potential_mv"],
            sham["max_abs_membrane_potential_mv"],
        ),
        "finite": stimulated["finite"] and sham["finite"],
    }


def _summarize(rows: list[dict]) -> dict:
    metrics = [key for key in rows[0] if key not in {"seed", "finite"}]
    return {
        metric: {
            "median": median(row[metric] for row in rows),
            "mean": fmean(row[metric] for row in rows),
        }
        for metric in metrics
    }


def run_biophysics(
    spec_path: Path,
    network_path: Path,
    channel_path: Path,
    output_path: Path,
    experiment_id: str | None = None,
) -> dict:
    spec = ExperimentSpec.load(spec_path)
    protocol = spec.biophysics_for(experiment_id)
    model = load_neuromuscular_neuroml(network_path, channel_path)
    if protocol.timestep_source != "network.recommended_dt_ms":
        raise ValueError(f"unsupported timestep source: {protocol.timestep_source}")
    if protocol.duration_source != "network.recommended_duration_ms":
        raise ValueError(f"unsupported duration source: {protocol.duration_source}")
    if protocol.synapse_model != "neuroml_event_conductance":
        raise ValueError(f"unsupported synapse model: {protocol.synapse_model}")
    if protocol.body_model != "damped_segment_chain":
        raise ValueError(f"unsupported body model: {protocol.body_model}")
    variants = build_biophysical_controls(
        model, protocol.controls, spec.seed, spec.rewiring_swaps_per_edge
    )
    arms = {
        name: [
            _run_pair(variant, protocol, seed, feedback_enabled)
            for seed in protocol.seeds
        ]
        for name, (variant, feedback_enabled) in variants.items()
    }
    summaries = {name: _summarize(rows) for name, rows in arms.items()}
    primary = protocol.primary_metric
    actual_name = "actual_closed_loop"
    controls = [name for name in variants if name != actual_name]
    pairwise_wins = {
        control: sum(
            actual[primary] > comparison[primary]
            for actual, comparison in zip(arms[actual_name], arms[control], strict=True)
        )
        for control in controls
    }
    actual_median = summaries[actual_name][primary]["median"]
    best_control_median = max(
        summaries[control][primary]["median"] for control in controls
    )
    all_finite = all(row["finite"] for rows in arms.values() for row in rows)
    landing_passed = (
        actual_median > best_control_median
        and all(
            wins >= protocol.minimum_pairwise_wins for wins in pairwise_wins.values()
        )
        and all_finite
    )
    cell_types = {cell.cell_id: cell.cell_type for cell in model.cells}
    neural_edges = [
        edge
        for edge in model.connections
        if cell_types[edge.source] != "muscle" and cell_types[edge.target] != "muscle"
    ]
    neuromuscular_edges = [
        edge for edge in model.connections if cell_types[edge.target] == "muscle"
    ]
    results = {
        "experiment_id": protocol.experiment_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment_id": spec.experiment_id,
        "source": {
            "repository": spec.source.repository,
            "revision": spec.source.revision,
            "model_path": spec.source.model_path,
            "sha256": spec.source.sha256,
            "include_files": [
                {
                    "model_path": artifact.model_path,
                    "sha256": artifact.sha256,
                }
                for artifact in spec.source.include_files
            ],
        },
        "protocol": {
            "seeds": list(protocol.seeds),
            "controls": list(protocol.controls),
            "timestep_ms": model.recommended_timestep_ms,
            "duration_ms": model.recommended_duration_ms,
            "stimulus_component_id": protocol.stimulus_component_id,
            "stimulus_targets": list(
                next(
                    stimulus.target_cell_ids
                    for stimulus in model.stimuli
                    if stimulus.stimulus_id == protocol.stimulus_component_id
                )
            ),
            "synapse_model": protocol.synapse_model,
            "body_model": protocol.body_model,
            "body_segments": protocol.body_segments,
            "primary_metric": primary,
            "minimum_pairwise_wins": protocol.minimum_pairwise_wins,
        },
        "manifest": {
            "cells": len(model.cells),
            "neurons": sum(cell.cell_type != "muscle" for cell in model.cells),
            "muscles": sum(cell.cell_type == "muscle" for cell in model.cells),
            "neural_connections": len(neural_edges),
            "neuromuscular_connections": len(neuromuscular_edges),
            "cell_components": [
                component.component_id for component in model.cell_components
            ],
            "ion_channels": [channel.channel_id for channel in model.ion_channels],
            "synapses": {
                mechanism.mechanism_id: sum(
                    edge.synapse == mechanism.mechanism_id for edge in model.connections
                )
                for mechanism in model.synapse_mechanisms
            },
        },
        "arms": arms,
        "summaries": summaries,
        "verdict": {
            "actual_median": actual_median,
            "best_control_median": best_control_median,
            "actual_to_best_control_ratio": actual_median
            / max(abs(best_control_median), 1e-12),
            "pairwise_seed_wins": pairwise_wins,
            "all_finite": all_finite,
            "landing_passed": landing_passed,
        },
    }
    if not all(
        math.isfinite(value["mean"])
        for summary in summaries.values()
        for value in summary.values()
    ):
        raise ValueError("biophysical dynamics produced a non-finite summary")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
