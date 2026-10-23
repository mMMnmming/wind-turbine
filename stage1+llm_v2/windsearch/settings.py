"""One explicit v2 configuration; paths resolve relative to the repository."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .client import DeepSeekSettings
from .domain import DeploymentConfig

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


@dataclass(frozen=True)
class SearchSettings:
    target_input: Path = ROOT / "turbine102.tsv"
    protected_input: Path = ROOT / "turbine87.tsv"
    data_dir: Path = LOCAL / "outputs/prepared"
    validation_inputs: tuple = ()
    training_virtual_fields: int = 8
    validation_virtual_fields: int = 8
    data_seed: int = 1234
    validation_seed: int = 1001234
    stage2_checkpoint: Path = ROOT / "stage2/outputs/finetune-edge-v2/control/best_greedy.pt"
    stage2_device: str = "cuda"
    gust_speed_mps: float = 12.0
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    evaluation_budget: int = 6
    max_attempts_per_round: int = 3
    seed: int = 1234
    search_seconds: float = 120.0
    max_proposals: int = 20000
    max_stage2_inferences: int = 5000
    stagnation_limit: int = 1000
    heartbeat_seconds: float = 15.0
    stage2_timeout_seconds: float = 300.0
    archive_size: int = 6

    def __post_init__(self):
        for key in ("evaluation_budget", "max_proposals", "max_stage2_inferences",
                    "stagnation_limit", "archive_size", "max_attempts_per_round"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + " must be a positive integer")
        if self.max_attempts_per_round > 3:
            raise ValueError("max_attempts_per_round must be 1..3 (initial, repair, restart)")
        for key in ("training_virtual_fields", "validation_virtual_fields", "seed",
                    "data_seed", "validation_seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(key + " must be a nonnegative integer")
        for key in ("search_seconds", "heartbeat_seconds", "stage2_timeout_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive")
        if self.stage2_device not in ("cpu", "cuda"):
            raise ValueError("stage2_device must be cpu or cuda")
        if not math.isfinite(self.gust_speed_mps) or self.gust_speed_mps < 0:
            raise ValueError("gust_speed_mps must be finite and nonnegative")

    def snapshot(self):
        return json.loads(json.dumps(asdict(self), default=str))


def load_settings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - set(SearchSettings.__dataclass_fields__)
    if unknown:
        raise ValueError("Unknown search settings: " + str(sorted(unknown)))
    for key in ("target_input", "protected_input", "data_dir", "stage2_checkpoint"):
        if key in raw:
            raw[key] = resolve(raw[key])
    raw["validation_inputs"] = tuple(resolve(p) for p in raw.get("validation_inputs", []))
    if "deployment" in raw:
        raw["deployment"] = DeploymentConfig(**raw["deployment"])
    return SearchSettings(**raw)


def load_deepseek_settings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) - set(DeepSeekSettings.__dataclass_fields__):
        raise ValueError("Unknown DeepSeek settings")
    # Attempts are explicit in the controller; never hide requests in SDK retries.
    raw["retries"] = 0
    return DeepSeekSettings(**raw)
