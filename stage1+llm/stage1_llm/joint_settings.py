"""Separate v5 configuration; frozen geometry continues using the v1 data manifest."""

from dataclasses import dataclass
import json
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

    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("stage2_device must be cpu or cuda")
        environment(self.search.deployment, self.gust_speed_mps)


def load_joint_settings(path: Path) -> JointSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(raw.pop("stage2_checkpoint", str(DEFAULT_CHECKPOINT)))
    device = raw.pop("stage2_device", "cuda")
    gust = raw.pop("gust_speed_mps", 12.0)
    version = raw.pop("search_version", 5)
    if version != 5:
        raise ValueError("Expected search_version=5")
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    return JointSettings(
        search_settings_from_dict(raw), checkpoint.resolve(), device, gust
    )
