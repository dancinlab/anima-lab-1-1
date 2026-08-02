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
    dynamics: DynamicsSpec

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            experiment_id=raw["experiment_id"],
            seed=int(raw["seed"]),
            variants=tuple(raw["variants"]),
            rewiring_swaps_per_edge=float(raw["rewiring_swaps_per_edge"]),
            source=SourceSpec(**raw["source"]),
            runtime=RuntimeSpec(**raw["runtime"]),
            dynamics=DynamicsSpec(
                **{
                    **raw["dynamics"],
                    "seeds": tuple(int(seed) for seed in raw["dynamics"]["seeds"]),
                }
            ),
        )

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
