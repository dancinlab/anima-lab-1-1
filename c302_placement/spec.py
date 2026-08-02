"""Experiment SSOT loader and pinned-source fetcher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class SourceSpec:
    repository: str
    revision: str
    model_path: str
    sha256: str
    neuron_component_contains: str

    @property
    def download_url(self) -> str:
        repository_path = self.repository.removeprefix(
            "https://github.com/"
        ).removesuffix(".git")
        return f"https://raw.githubusercontent.com/{repository_path}/{self.revision}/{self.model_path}"


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
class ExperimentSpec:
    experiment_id: str
    seed: int
    variants: tuple[str, ...]
    rewiring_swaps_per_edge: float
    source: SourceSpec
    runtime: RuntimeSpec
    default_dynamics_experiment_id: str
    dynamics_experiments: tuple[DynamicsSpec, ...]

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
        spec = cls(
            experiment_id=raw["experiment_id"],
            seed=int(raw["seed"]),
            variants=tuple(raw["variants"]),
            rewiring_swaps_per_edge=float(raw["rewiring_swaps_per_edge"]),
            source=SourceSpec(**raw["source"]),
            runtime=RuntimeSpec(**raw["runtime"]),
            default_dynamics_experiment_id=dynamics[
                "default_experiment_id"
            ],
            dynamics_experiments=experiments,
        )
        spec.dynamics_for()
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
