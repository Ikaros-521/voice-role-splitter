"""Run configuration: one YAML file per labelling run, everything else derived from paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised for a config a user can fix; the message always says what to change."""


DEFAULT_ROLES = [
    ["unknown", "— 未确认 —"],
]


@dataclass
class Config:
    audio_dir: str
    run_dir: str = "runs/default"
    sample_rate: int = 16000
    window: float = 3.0
    hop: float = 1.5
    min_speech: float = 0.6          # trimmed audio shorter than this gets no embedding
    trim_db: float = 35.0
    cluster_min_speech: float = 1.0   # and is kept out of clustering below this
    models: list = field(default_factory=lambda: ["campplus", "ecapa"])
    model_dir: str = ""               # local weight cache; empty = tool default
    device: str = "auto"
    linkage: str = "average"
    score: str = "asnorm"             # asnorm | cosine
    center: bool = True
    ks: list = field(default_factory=lambda: list(range(5, 31)))
    min_cluster: int = 25             # clusters below this are folded into the nearest one
    # re-scoring
    k_neighbors: int = 12
    silver_pct: float = 0.55
    keep_margin: float = 0.03
    assign_margin: float = 0.03
    only_unassigned: bool = True
    # annotation UI
    roles: list = field(default_factory=lambda: list(DEFAULT_ROLES))
    per_cluster: int = 6
    spot: int = 3
    seed: int = 0

    # ---------- paths ----------
    @property
    def root(self) -> Path:
        return Path(self.run_dir).expanduser().resolve()

    def path(self, *parts: str) -> Path:
        """file inside the run dir, creating parent folders"""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- io ----------
    @classmethod
    def load(cls, run_dir: str) -> "Config":
        f = Path(run_dir) / "run.yaml"
        if not f.exists():
            raise ConfigError(f'no run.yaml in "{run_dir}" -- create it with: vrs init --audio-dir DIR')
        data = yaml.safe_load(f.read_text("utf-8")) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        cfg = cls(**known)
        cfg._unknown_keys = sorted(unknown)
        cfg.validate()
        return cfg

    def save(self, run_dir: str | None = None) -> Path:
        if run_dir:
            self.run_dir = run_dir
        d = Path(self.run_dir)
        d.mkdir(parents=True, exist_ok=True)
        f = d / "run.yaml"
        f.write_text(yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False), "utf-8")
        return f

    def validate(self) -> None:
        a = Path(self.audio_dir)
        if not a.exists():
            raise ConfigError(f'audio_dir does not exist: "{a}" (fix run.yaml or pass --audio-dir)')
        if not any(a.glob("*.wav")):
            raise ConfigError(f'no .wav files in audio_dir "{a}" -- vrs currently reads WAV only')
        if self.window <= self.hop:
            raise ConfigError("window must be greater than hop, otherwise windows never advance")
        if self.min_speech <= 0 or self.sample_rate < 8000:
            raise ConfigError("min_speech must be > 0 and sample_rate >= 8000")
        if self.score not in ("asnorm", "cosine"):
            raise ConfigError(f'score must be asnorm or cosine, got "{self.score}"')
        if self.linkage not in ("average", "complete", "single"):
            raise ConfigError(f'linkage must be average/complete/single, got "{self.linkage}"')
        if not self.ks:
            raise ConfigError("ks is empty: nothing to sweep -- set ks: [8, 12, 16] for example")
        if not isinstance(self.models, (list, tuple)) or not self.models:
            raise ConfigError("models must be a non-empty list, e.g. [campplus, ecapa] -- "
                              "run `vrs backends` to see what is installed")
        if not isinstance(self.ks, (list, tuple)) or any(int(k) < 2 for k in self.ks):
            raise ConfigError(f"every ks entry must be >= 2, got {self.ks}")
        if not isinstance(self.roles, list) or not self.roles:
            raise ConfigError("roles must be a non-empty list of [value, label] pairs for the page")
        if getattr(self, "_unknown_keys", None):
            raise ConfigError(f"unknown key(s) in run.yaml: {', '.join(self._unknown_keys)}")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def resolved_model_dir(self) -> Path:
        """weights live outside the run dir: speechbrain symlinks into its savedir, so a
        weights folder inside a run makes the run un-copyable, and runs should be copyable."""
        if self.model_dir:
            return Path(self.model_dir)
        env = os.environ.get("VRS_MODEL_DIR")
        if env:
            return Path(env)
        return Path.home() / ".cache" / "vrs" / "models"
