"""Refresh whole synthetic fields without changing fixed validation problems."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .atsp import WindATSPInstance
from .data import build_kde_instances, instance_fingerprint, instance_from_record, instance_to_record, split_fingerprint


def source_hashes(turbine_path: Path, config_path: Path) -> dict[str, str]:
    """Check content rather than absolute paths so prepared data is portable."""
    return {
        "turbine_tsv_sha256": hashlib.sha256(turbine_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "stage1_config_sha256": hashlib.sha256(config_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    }


def field_id(instance: WindATSPInstance) -> int | None:
    """Synthetic groups from one farm share the virtual field identifier."""
    name = instance.identifiers[instance.depot_index]
    return int(name.split("-")[1]) if name.startswith("virtual-") else None


def assert_disjoint(train: list[WindATSPInstance], validation: list[WindATSPInstance]) -> None:
    """Reject identical routing problems across data splits."""
    if {instance_fingerprint(item) for item in train} & {instance_fingerprint(item) for item in validation}:
        raise ValueError("Training and validation problems overlap.")


def sample_training_indices(
    instances: list[WindATSPInstance], batch_size: int, generator: np.random.Generator,
    stratified: bool = False,
) -> np.ndarray:
    """Sample unique graphs, optionally balancing ten-client size buckets."""
    if not instances or batch_size < 1:
        raise ValueError("Training requires instances and a positive batch size.")
    count = min(batch_size, len(instances))
    if not stratified:
        return generator.choice(len(instances), size=count, replace=False)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        buckets[(instance.node_count - 2) // 10].append(index)
    queues = [generator.permutation(indices).tolist() for indices in buckets.values()]
    selected = []
    while len(selected) < count:
        for bucket_index in generator.permutation(len(queues)):
            queue = queues[int(bucket_index)]
            if queue:
                selected.append(queue.pop())
                if len(selected) == count:
                    break
    return generator.permutation(selected)


class TrainingPool:
    """FIFO replacement of complete KDE fields, with checkpointable state."""

    def __init__(
        self, instances: list[WindATSPInstance], validation_instances: list[WindATSPInstance],
        turbine_path: Path, config_path: Path, seed: int = 1234,
        refresh_every_epochs: int = 0, refresh_fields: int = 0, max_attempts_per_field: int = 20,
    ) -> None:
        assert_disjoint(instances, validation_instances)
        self.instances = list(instances)
        self.validation_instances = list(validation_instances)
        self.turbine_path = turbine_path
        self.config_path = config_path
        self.seed = seed
        self.refresh_every_epochs = refresh_every_epochs
        self.refresh_fields = refresh_fields
        self.max_attempts_per_field = max_attempts_per_field
        ids = sorted({field_id(item) for item in instances} - {None})
        self.capacity = len(ids)
        if not instances or not validation_instances:
            raise ValueError("Training and validation splits must not be empty.")
        if refresh_every_epochs < 0 or refresh_fields < 0 or seed < 0:
            raise ValueError("Pool seeds and refresh settings must be nonnegative.")
        if refresh_every_epochs and not 1 <= refresh_fields <= self.capacity:
            raise ValueError("refresh_fields must be between one and the initial KDE field count.")
        if ids and (ids[0] < 0 or ids[-1] >= 1_000_000):
            raise ValueError("Training field IDs must stay below the validation stream at 1000000.")
        self.next_field_id = ids[-1] + 1 if ids else 0
        self.version = 0
        self.last_refresh_epoch = 0
        self.generation_records: list[dict[str, object]] = []
        self.input_hashes = source_hashes(turbine_path, config_path)
        self.initial_fingerprint = split_fingerprint(instances)
        self.real_fingerprints = [instance_fingerprint(item) for item in instances if field_id(item) is None]

    def _settings(self) -> dict[str, object]:
        return {
            "seed": self.seed, "capacity": self.capacity, "refresh_every_epochs": self.refresh_every_epochs,
            "refresh_fields": self.refresh_fields, "max_attempts_per_field": self.max_attempts_per_field,
            "input_hashes": self.input_hashes, "initial_fingerprint": self.initial_fingerprint,
            "validation_fingerprint": split_fingerprint(self.validation_instances),
        }

    def maybe_refresh(self, epoch_index: int) -> bool:
        """Refresh before epochs 6, 11, ... when the interval is five."""
        if (not self.refresh_every_epochs or epoch_index <= self.last_refresh_epoch
                or epoch_index % self.refresh_every_epochs):
            return False
        if self.next_field_id + self.refresh_fields > 1_000_000:
            raise ValueError("Training seed stream exhausted; refusing to enter the validation seed stream.")
        if source_hashes(self.turbine_path, self.config_path) != self.input_hashes:
            raise ValueError("Source files changed while refreshing the training pool.")
        records: list[dict[str, object]] = []
        fresh = build_kde_instances(
            self.turbine_path, self.config_path, self.refresh_fields, self.seed + self.next_field_id,
            field_offset=self.next_field_id, gust_speed_mps=self.instances[0].config.gust_speed_mps,
            max_attempts_per_field=self.max_attempts_per_field, generation_records=records,
        )
        old_ids = sorted({field_id(item) for item in self.instances} - {None})
        evicted = set(old_ids[:self.refresh_fields])
        retained = [item for item in self.instances if field_id(item) not in evicted]
        replacement = [*retained, *fresh]
        assert_disjoint(replacement, self.validation_instances)
        if len({field_id(item) for item in fresh}) != self.refresh_fields:
            raise ValueError("Refresh did not produce the requested number of complete fields.")
        self.instances = replacement
        self.generation_records = records
        self.next_field_id += self.refresh_fields
        self.version += 1
        self.last_refresh_epoch = epoch_index
        return True

    def state_dict(self) -> dict[str, object]:
        """Include active geometry, seeds, and provenance, not just a counter."""
        return {
            "schema_version": 1, "settings": self._settings(), "version": self.version,
            "next_field_id": self.next_field_id, "last_refresh_epoch": self.last_refresh_epoch,
            "instances": [instance_to_record(item) for item in self.instances],
            "active_fingerprint": split_fingerprint(self.instances), "generation_records": self.generation_records,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Reject changed data/settings rather than silently restart a stream."""
        if state.get("schema_version") != 1 or state["settings"] != self._settings():
            raise ValueError("Checkpoint training-pool settings differ from the current data/settings.")
        instances = [instance_from_record(record) for record in state["instances"]]
        if split_fingerprint(instances) != state["active_fingerprint"]:
            raise ValueError("Checkpoint pool fingerprint is invalid.")
        real = [instance_fingerprint(item) for item in instances if field_id(item) is None]
        if real != self.real_fingerprints or len({field_id(item) for item in instances} - {None}) != self.capacity:
            raise ValueError("Checkpoint pool has changed real regions or field capacity.")
        assert_disjoint(instances, self.validation_instances)
        self.instances = instances
        self.version = int(state["version"])
        self.next_field_id = int(state["next_field_id"])
        self.last_refresh_epoch = int(state["last_refresh_epoch"])
        self.generation_records = list(state["generation_records"])

    def write_snapshot(self, output: Path) -> None:
        """Keep immutable versioned pool manifests separate from initial data."""
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"pool_{self.version:04d}.json"
        content = json.dumps(self.state_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Existing pool snapshot differs: {path}. Use a fresh output directory.")
        else:
            path.write_text(content, encoding="utf-8")
