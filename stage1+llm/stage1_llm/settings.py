"""Strict configuration loading; paths resolve from the repository root."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from .domain import DeploymentConfig
from .client import DeepSeekSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTION_ACTIONS = {
    "constraint_repair": "Repair an explicit hard-constraint failure.",
    "reduce_nests": "Reduce nest count while retaining every hard constraint.",
    "rebalance_assignments": "Improve assignment and partition balance.",
    "increase_safety_margin": "Improve endurance or service-radius safety margin.",
    "reduce_route_distance": "Reduce total directed proxy flight time without relaxing conservative feasibility.",
    "structural_exploration": "Explore a substantially different deployment structure.",
    "simplify_tune": "Remove redundant logic or tune internal algorithm constants.",
}
DEFAULT_DIAGNOSTIC_DIRECTION_MAP = {
    "coverage": "constraint_repair",
    "undeclared_nest": "constraint_repair",
    "invalid_nest": "constraint_repair",
    "endurance": "increase_safety_margin",
    "service_radius": "increase_safety_margin",
}


@dataclass(frozen=True)
class SearchSettings:
    target_input: Path
    validation_inputs: tuple[Path, ...]
    deployment: DeploymentConfig
    archive_size: int = 12
    initial_candidates: int = 3
    evaluation_budget: int = 40
    no_improvement_limit: int = 10
    seed: int = 1234
    candidate_timeout_seconds: int = 300
    ucb_decay: float = 0.9
    ucb_exploration: float = 0.7
    lower_bound_nests: int = 1
    direction_actions: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DIRECTION_ACTIONS)
    )
    diagnostic_direction_map: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DIAGNOSTIC_DIRECTION_MAP)
    )
    data_dir: Path = REPO_ROOT / "stage1+llm" / "outputs" / "prepared-v3"
    protected_input: Path = REPO_ROOT / "turbine87.tsv"
    training_virtual_fields: int = 8
    validation_virtual_fields: int = 8
    data_seed: int = 1234
    validation_seed: int = 1001234
    heartbeat_seconds: float = 10.0
    show_code: bool = True
    selector: str = "discounted"
    ucb_window: int = 32
    constraint_preflight: bool = False
    early_rejection: bool = False
    soft_seconds: float = 120.0
    max_region_evaluations: int = 5000
    max_move_attempts: int = 20000
    feedback_dataset_limit: int = 4

    def __post_init__(self) -> None:
        counts = (
            self.archive_size,
            self.initial_candidates,
            self.evaluation_budget,
            self.no_improvement_limit,
            self.candidate_timeout_seconds,
        )
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Search counts and timeout must be positive integers.")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.training_virtual_fields,
                self.validation_virtual_fields,
                self.data_seed,
                self.validation_seed,
            )
        ):
            raise ValueError("Topology counts and seeds must be nonnegative integers.")
        import math

        if not math.isfinite(self.heartbeat_seconds) or self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be finite and positive.")
        if type(self.show_code) is not bool:
            raise ValueError("show_code must be a boolean.")
        if self.selector not in ("discounted", "sliding_window"):
            raise ValueError("Unknown selector.")
        if any(
            type(v) is not int or v < 1
            for v in (
                self.ucb_window,
                self.max_region_evaluations,
                self.max_move_attempts,
                self.feedback_dataset_limit,
            )
        ):
            raise ValueError("Work/window/feedback limits must be positive integers.")
        if not math.isfinite(self.soft_seconds) or self.soft_seconds <= 0:
            raise ValueError(
                "soft_seconds must be positive and finite (runner clamps to hard timeout)."
            )
        if any(
            type(v) is not bool
            for v in (self.constraint_preflight, self.early_rejection)
        ):
            raise ValueError("Preflight/early rejection switches must be booleans.")
        if (
            not 0 < self.ucb_decay <= 1
            or not math.isfinite(self.ucb_exploration)
            or self.ucb_exploration < 0
        ):
            raise ValueError("Invalid UCB configuration.")


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Config file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON config: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_search_settings(
    path: Path, validation_overrides: list[str] | None = None
) -> SearchSettings:
    raw = _load_json(path)
    return search_settings_from_dict(raw, validation_overrides)


def search_settings_from_dict(raw: dict, validation_overrides=None) -> SearchSettings:
    """Share strict legacy field validation with the v5 configuration wrapper."""
    allowed = set(SearchSettings.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown search config keys: {sorted(unknown)}")
    if not isinstance(raw.get("deployment"), dict):
        raise ValueError("deployment must be a JSON object.")
    deployment = DeploymentConfig(**raw["deployment"])
    target = _resolve(raw.get("target_input", "turbine102.tsv"))
    values = list(raw.get("validation_inputs", [])) + list(validation_overrides or [])
    validations = tuple(dict.fromkeys(_resolve(p) for p in values))
    extra = {
        k: v
        for k, v in raw.items()
        if k not in {"target_input", "validation_inputs", "deployment"}
    }
    for name in ("data_dir", "protected_input"):
        if name in extra:
            extra[name] = _resolve(extra[name])
    settings = SearchSettings(target, validations, deployment, **extra)
    if (
        min(
            settings.archive_size,
            settings.initial_candidates,
            settings.evaluation_budget,
            settings.no_improvement_limit,
            settings.candidate_timeout_seconds,
        )
        < 1
    ):
        raise ValueError("Search counts and timeout must be positive.")
    if settings.lower_bound_nests != 1:
        raise ValueError(
            "Only the universally certified non-empty-instance lower bound of one nest is supported currently."
        )
    if set(settings.direction_actions) != set(DEFAULT_DIRECTION_ACTIONS) or not all(
        isinstance(value, str) and value.strip()
        for value in settings.direction_actions.values()
    ):
        raise ValueError(
            "direction_actions must define every supported direction with a non-empty description."
        )
    if not set(settings.diagnostic_direction_map).issubset(
        DEFAULT_DIAGNOSTIC_DIRECTION_MAP
    ) or any(
        value not in settings.direction_actions
        for value in settings.diagnostic_direction_map.values()
    ):
        raise ValueError(
            "diagnostic_direction_map contains an unsupported diagnostic or direction."
        )
    return settings


def load_deepseek_settings(path: Path) -> DeepSeekSettings:
    raw = _load_json(path)
    allowed = set(DeepSeekSettings.__dataclass_fields__)
    if set(raw) - allowed:
        raise ValueError(f"Unknown DeepSeek config keys: {sorted(set(raw)-allowed)}")
    return DeepSeekSettings(**raw)
