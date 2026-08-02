"""Pre-registered named-neuron runtime dynamics experiment."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev

from .controls import build_variants
from .model import Connectome
from .neuroml import load_neuroml
from .runtime import bind_connectome, connection_length_scale
from .spec import DynamicsSpec, ExperimentSpec


def _role_indices(
    connectome: Connectome,
    include_roles: tuple[str, ...],
    exclude_roles: tuple[str, ...],
) -> list[int]:
    include = {role.lower() for role in include_roles}
    exclude = {role.lower() for role in exclude_roles}
    if not include:
        raise ValueError("role selection requires at least one included role")
    indices = []
    for index, neuron in enumerate(connectome.neurons):
        roles = {
            part.strip() for part in neuron.neuron_type.lower().split(";")
        }
        if include.issubset(roles) and roles.isdisjoint(exclude):
            indices.append(index)
    if not indices:
        raise ValueError(
            f"role selection contains no neurons: include={sorted(include)}, "
            f"exclude={sorted(exclude)}"
        )
    return indices


def _response_rms(delta, indices: list[int]) -> float:
    if not indices:
        raise ValueError("response role contains no neurons")
    return float(delta[indices].square().mean().sqrt().item())


def _run_arm(
    connectome: Connectome,
    reference_distance_scale: float,
    spec: ExperimentSpec,
    dynamics: DynamicsSpec,
    seed: int,
) -> dict:
    import torch

    from consciousness_engine import ConsciousnessEngine

    torch.manual_seed(seed)
    engine = ConsciousnessEngine(
        cell_dim=dynamics.cell_dim,
        hidden_dim=dynamics.hidden_dim,
        initial_cells=len(connectome.neurons),
        max_cells=len(connectome.neurons),
        phi_ratchet=dynamics.phi_ratchet,
    )
    bind_connectome(
        engine,
        connectome,
        spec.runtime.coupling_normalization,
        spec.runtime.lock_structure,
        lock_population=spec.runtime.lock_population,
        spatial_kernel=dynamics.spatial_kernel,
        distance_scale=reference_distance_scale,
        synapse_model=dynamics.synapse_model,
        runtime_timestep_ms=dynamics.runtime_timestep_ms,
    )

    zero = torch.zeros(len(connectome.neurons), dynamics.cell_dim)
    for _ in range(dynamics.warmup_steps):
        engine.step(cell_inputs=zero)

    stimulated = copy.deepcopy(engine)
    sham = copy.deepcopy(engine)
    stimulus_indices = _role_indices(
        connectome,
        dynamics.stimulus_include_roles,
        dynamics.stimulus_exclude_roles,
    )
    readout_indices = _role_indices(
        connectome,
        dynamics.readout_include_roles,
        dynamics.readout_exclude_roles,
    )

    generator = torch.Generator().manual_seed(seed)
    stimulus_vector = (
        torch.randint(0, 2, (dynamics.cell_dim,), generator=generator).float() * 2 - 1
    )
    stimulus_vector = (
        stimulus_vector / stimulus_vector.norm() * dynamics.stimulus_amplitude
    )
    stimulus = zero.clone()
    stimulus[stimulus_indices] = stimulus_vector

    motor_trace: list[float] = []
    sensory_trace: list[float] = []
    phi_delta_trace: list[float] = []
    tension_delta_trace: list[float] = []
    sham_tension_trace: list[float] = []
    population_preserved = True
    total_steps = dynamics.stimulus_steps + dynamics.recovery_steps
    for step in range(total_steps):
        active_input = stimulus if step < dynamics.stimulus_steps else zero
        stimulated_result = stimulated.step(cell_inputs=active_input)
        sham_result = sham.step(cell_inputs=zero)
        delta = stimulated_result["cell_outputs"] - sham_result["cell_outputs"]
        motor_trace.append(_response_rms(delta, readout_indices))
        sensory_trace.append(_response_rms(delta, stimulus_indices))
        phi_delta_trace.append(
            float(stimulated_result["phi_iit"] - sham_result["phi_iit"])
        )
        stimulated_tension = torch.tensor(stimulated_result["tensions"])
        sham_tension = torch.tensor(sham_result["tensions"])
        tension_delta_trace.append(
            float((stimulated_tension - sham_tension).abs().mean().item())
        )
        sham_tension_trace.append(float(sham_tension.mean().item()))
        population_preserved &= (
            stimulated_result["n_cells"] == len(connectome.neurons)
            and sham_result["n_cells"] == len(connectome.neurons)
        )

    sham_tension_mean = fmean(sham_tension_trace)
    return {
        "seed": seed,
        "stimulus_neurons": len(stimulus_indices),
        "readout_neurons": len(readout_indices),
        "sensory_neurons": len(stimulus_indices),
        "motor_neurons": len(readout_indices),
        "readout_response_auc": sum(motor_trace),
        "readout_response_peak": max(motor_trace),
        "stimulus_response_auc": sum(sensory_trace),
        "readout_stimulus_transmission": sum(motor_trace)
        / max(sum(sensory_trace), 1e-12),
        "motor_response_auc": sum(motor_trace),
        "motor_response_peak": max(motor_trace),
        "sensory_response_auc": sum(sensory_trace),
        "motor_sensory_transmission": sum(motor_trace)
        / max(sum(sensory_trace), 1e-12),
        "phi_signed_delta_auc": sum(phi_delta_trace),
        "phi_absolute_delta_auc": sum(abs(value) for value in phi_delta_trace),
        "tension_delta_auc": sum(tension_delta_trace),
        "sham_tension_cv": pstdev(sham_tension_trace)
        / max(abs(sham_tension_mean), 1e-12),
        "population_preserved": population_preserved,
    }


def _summarize(seed_results: list[dict]) -> dict:
    metric_names = (
        "motor_response_auc",
        "motor_response_peak",
        "sensory_response_auc",
        "motor_sensory_transmission",
        "phi_signed_delta_auc",
        "phi_absolute_delta_auc",
        "tension_delta_auc",
        "sham_tension_cv",
        "readout_response_auc",
        "readout_response_peak",
        "stimulus_response_auc",
        "readout_stimulus_transmission",
    )
    return {
        metric: {
            "median": median(result[metric] for result in seed_results),
            "mean": fmean(result[metric] for result in seed_results),
        }
        for metric in metric_names
    }


def _synapse_manifest(connectome: Connectome) -> list[dict]:
    counts: dict[str, int] = {}
    for edge in connectome.connections:
        counts[edge.synapse] = counts.get(edge.synapse, 0) + 1
    return [
        {
            "mechanism_id": mechanism.mechanism_id,
            "kind": mechanism.kind,
            "connection_count": counts[mechanism.mechanism_id],
            "reversal_potential_mv": mechanism.reversal_potential_mv,
            "rise_time_ms": mechanism.rise_time_ms,
            "decay_time_ms": mechanism.decay_time_ms,
        }
        for mechanism in connectome.synapse_mechanisms
        if mechanism.mechanism_id in counts
    ]


def run_dynamics(
    spec_path: Path,
    source_path: Path,
    output_path: Path,
    experiment_id: str | None = None,
) -> dict:
    spec = ExperimentSpec.load(spec_path)
    dynamics = spec.dynamics_for(experiment_id)
    if dynamics.distance_scale != "actual_median_connection_length":
        raise ValueError(
            f"unsupported distance scale: {dynamics.distance_scale}"
        )
    actual = load_neuroml(source_path, spec.source.neuron_component_contains)
    variants = build_variants(
        actual, spec.variants, spec.seed, spec.rewiring_swaps_per_edge
    )
    distance_scale = connection_length_scale(actual)
    arms = {
        name: [
            _run_arm(variant, distance_scale, spec, dynamics, seed)
            for seed in dynamics.seeds
        ]
        for name, variant in variants.items()
    }
    summaries = {name: _summarize(results) for name, results in arms.items()}
    primary = dynamics.primary_metric
    controls = [name for name in variants if name != "actual"]
    pairwise_wins = {
        control: sum(
            actual_result[primary] > control_result[primary]
            for actual_result, control_result in zip(
                arms["actual"], arms[control], strict=True
            )
        )
        for control in controls
    }
    actual_median = summaries["actual"][primary]["median"]
    best_control_median = max(
        summaries[control][primary]["median"] for control in controls
    )
    landing_passed = (
        actual_median > best_control_median
        and all(
            wins >= dynamics.minimum_pairwise_wins
            for wins in pairwise_wins.values()
        )
        and all(
            result["population_preserved"]
            for seed_results in arms.values()
            for result in seed_results
        )
    )
    results = {
        "experiment_id": dynamics.experiment_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment_id": spec.experiment_id,
        "source": {
            "repository": spec.source.repository,
            "revision": spec.source.revision,
            "model_path": spec.source.model_path,
            "sha256": spec.source.sha256,
        },
        "protocol": {
            "seeds": list(dynamics.seeds),
            "cell_dim": dynamics.cell_dim,
            "hidden_dim": dynamics.hidden_dim,
            "warmup_steps": dynamics.warmup_steps,
            "stimulus_steps": dynamics.stimulus_steps,
            "recovery_steps": dynamics.recovery_steps,
            "stimulus_amplitude": dynamics.stimulus_amplitude,
            "spatial_kernel": dynamics.spatial_kernel,
            "distance_scale_rule": dynamics.distance_scale,
            "distance_scale": distance_scale,
            "synapse_model": dynamics.synapse_model,
            "runtime_timestep_ms": dynamics.runtime_timestep_ms,
            "resting_potential_mv": actual.resting_potential_mv,
            "synapse_channels": _synapse_manifest(actual),
            "stimulus_include_roles": list(dynamics.stimulus_include_roles),
            "stimulus_exclude_roles": list(dynamics.stimulus_exclude_roles),
            "readout_include_roles": list(dynamics.readout_include_roles),
            "readout_exclude_roles": list(dynamics.readout_exclude_roles),
            "primary_metric": primary,
            "minimum_pairwise_wins": dynamics.minimum_pairwise_wins,
        },
        "arms": arms,
        "summaries": summaries,
        "verdict": {
            "actual_median": actual_median,
            "best_control_median": best_control_median,
            "actual_to_best_control_ratio": actual_median
            / max(best_control_median, 1e-12),
            "pairwise_seed_wins": pairwise_wins,
            "landing_passed": landing_passed,
        },
    }
    if not all(
        math.isfinite(value["mean"])
        for summary in summaries.values()
        for value in summary.values()
    ):
        raise ValueError("dynamics produced a non-finite summary")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
