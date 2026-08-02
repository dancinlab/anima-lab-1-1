"""Experiment SSOT loader and pinned-source fetcher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class SourceArtifactSpec:
    model_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    repository: str
    revision: str
    model_path: str
    sha256: str
    neuron_component_contains: str
    include_files: tuple[SourceArtifactSpec, ...] = ()

    @property
    def download_url(self) -> str:
        return self.artifact_url(self.model_path)

    def artifact_url(self, model_path: str) -> str:
        repository_path = self.repository.removeprefix(
            "https://github.com/"
        ).removesuffix(".git")
        return f"https://raw.githubusercontent.com/{repository_path}/{self.revision}/{model_path}"


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    coupling_normalization: str
    lock_structure: bool
    lock_population: bool


@dataclass(frozen=True, slots=True)
class DynamicsSpec:
    experiment_id: str
    result_path: str
    seeds: tuple[int, ...]
    cell_dim: int
    hidden_dim: int
    phi_ratchet: bool
    warmup_steps: int
    stimulus_steps: int
    recovery_steps: int
    stimulus_amplitude: float
    spatial_kernel: str
    distance_scale: str
    synapse_model: str
    runtime_timestep_ms: float
    stimulus_include_roles: tuple[str, ...]
    stimulus_exclude_roles: tuple[str, ...]
    readout_include_roles: tuple[str, ...]
    readout_exclude_roles: tuple[str, ...]
    primary_metric: str
    minimum_pairwise_wins: int


@dataclass(frozen=True, slots=True)
class BiophysicsSpec:
    experiment_id: str
    result_path: str
    seeds: tuple[int, ...]
    controls: tuple[str, ...]
    timestep_source: str
    duration_source: str
    stimulus_component_id: str
    initial_voltage_jitter_mv: float
    synapse_model: str
    body_model: str
    body_segments: int
    muscle_activation_slope_mv: float
    muscle_activation_time_constant_ms: float
    body_stiffness: float
    body_damping: float
    body_torque_gain: float
    proprioceptive_feedback_gain_pa: float
    primary_metric: str
    minimum_pairwise_wins: int


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    seed: int
    variants: tuple[str, ...]
    rewiring_swaps_per_edge: float
    source: SourceSpec
    runtime: RuntimeSpec
    default_dynamics_experiment_id: str
    dynamics_experiments: tuple[DynamicsSpec, ...]
    default_biophysics_experiment_id: str
    biophysics_experiments: tuple[BiophysicsSpec, ...]

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec:
        raw = json.loads(path.read_text(encoding="utf-8"))
        dynamics = raw["dynamics"]
        experiments = tuple(
            DynamicsSpec(
                **{
                    **experiment,
                    "seeds": tuple(int(seed) for seed in experiment["seeds"]),
                    "stimulus_include_roles": tuple(
                        experiment["stimulus_include_roles"]
                    ),
                    "stimulus_exclude_roles": tuple(
                        experiment["stimulus_exclude_roles"]
                    ),
                    "readout_include_roles": tuple(
                        experiment["readout_include_roles"]
                    ),
                    "readout_exclude_roles": tuple(
                        experiment["readout_exclude_roles"]
                    ),
                }
            )
            for experiment in dynamics["experiments"]
        )
        biophysics = raw["biophysics"]
        biophysics_experiments = tuple(
            BiophysicsSpec(
                **{
                    **experiment,
                    "seeds": tuple(int(seed) for seed in experiment["seeds"]),
                    "controls": tuple(experiment["controls"]),
                }
            )
            for experiment in biophysics["experiments"]
        )
        source = raw["source"]
        spec = cls(
            experiment_id=raw["experiment_id"],
            seed=int(raw["seed"]),
            variants=tuple(raw["variants"]),
            rewiring_swaps_per_edge=float(raw["rewiring_swaps_per_edge"]),
            source=SourceSpec(
                **{
                    **source,
                    "include_files": tuple(
                        SourceArtifactSpec(**artifact)
                        for artifact in source.get("include_files", [])
                    ),
                }
            ),
            runtime=RuntimeSpec(**raw["runtime"]),
            default_dynamics_experiment_id=dynamics[
                "default_experiment_id"
            ],
            dynamics_experiments=experiments,
            default_biophysics_experiment_id=biophysics[
                "default_experiment_id"
            ],
            biophysics_experiments=biophysics_experiments,
        )
        spec.dynamics_for()
        spec.biophysics_for()
        return spec

    @property
    def dynamics(self) -> DynamicsSpec:
        """Return the SSOT-selected default dynamics protocol."""
        return self.dynamics_for()

    def dynamics_for(self, experiment_id: str | None = None) -> DynamicsSpec:
        selected = experiment_id or self.default_dynamics_experiment_id
        matches = [
            experiment
            for experiment in self.dynamics_experiments
            if experiment.experiment_id == selected
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicate dynamics experiment: {selected}")
        return matches[0]

    @property
    def biophysics(self) -> BiophysicsSpec:
        return self.biophysics_for()

    def biophysics_for(self, experiment_id: str | None = None) -> BiophysicsSpec:
        selected = experiment_id or self.default_biophysics_experiment_id
        matches = [
            experiment
            for experiment in self.biophysics_experiments
            if experiment.experiment_id == selected
        ]
        if len(matches) != 1:
            raise ValueError(
                f"unknown or duplicate biophysics experiment: {selected}"
            )
        return matches[0]

    def fetch(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with urlopen(self.source.download_url, timeout=60) as response:
                destination.write_bytes(response.read())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != self.source.sha256:
            raise ValueError(
                f"source checksum mismatch: expected {self.source.sha256}, got {digest}"
            )
        return destination
