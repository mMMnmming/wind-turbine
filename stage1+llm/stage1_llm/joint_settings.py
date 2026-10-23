"""Separate v5 configuration; frozen geometry continues using the v1 data manifest."""

import json
from dataclasses import dataclass
from pathlib import Path

from .joint_planner import DEFAULT_CHECKPOINT, REPO_ROOT, environment
from .settings import SearchSettings, search_settings_from_dict

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/search-v5.json"


@dataclass(frozen=True)
class JointSettings:
    search: SearchSettings
    checkpoint: Path = DEFAULT_CHECKPOINT
    device: str = "cuda"
    gust_speed_mps: float = 12.0
    search_mode: str = "standard"
    seed_candidate: Path | None = None

    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("stage2_device must be cpu or cuda")
        environment(self.search.deployment, self.gust_speed_mps)
        if self.search_mode not in ("standard", "focused_six"):
            raise ValueError("search_mode must be standard or focused_six")
        if self.search_mode == "focused_six" and (
            self.seed_candidate is None
            or self.search.evaluation_budget != 6
            or self.search.no_improvement_limit < 6
        ):
            raise ValueError(
                "focused_six requires an explicit seed, exactly 6 rounds and stall limit >=6"
            )
        if self.search_mode == "standard" and self.seed_candidate is not None:
            raise ValueError("seed_candidate is only supported in focused_six mode")


def load_joint_settings(path: Path) -> JointSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(raw.pop("stage2_checkpoint", str(DEFAULT_CHECKPOINT)))
    device = raw.pop("stage2_device", "cuda")
    gust = raw.pop("gust_speed_mps", 12.0)
    version = raw.pop("search_version", 5)
    mode = raw.pop("search_mode", "standard")
    seed = raw.pop("seed_candidate", None)
    if seed is not None:
        seed = Path(seed)
        seed = seed if seed.is_absolute() else REPO_ROOT / seed
    if version != 5:
        raise ValueError("Expected search_version=5")
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    return JointSettings(
        search_settings_from_dict(raw), checkpoint.resolve(), device, gust, mode, seed
    )
