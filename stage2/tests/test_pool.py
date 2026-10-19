"""Fixed-data integrity, deterministic pool refresh, and sampling tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stage2.atsp import WindATSPConfig, WindATSPInstance
from stage2.data import instance_fingerprint, instance_to_record, load_region_instances
from stage2.pool import TrainingPool, sample_training_indices, source_hashes
from stage2.run import _prepare_or_load_dataset


ROOT = Path(__file__).resolve().parents[2]


def small_instance(name: str, offset: float, nodes: int = 3) -> WindATSPInstance:
    coordinates = np.array([[offset + index * 0.1, index % 2 * 0.1] for index in range(nodes)])
    return WindATSPInstance(
        (name, *(f"{name}-client-{index}" for index in range(1, nodes))),
        coordinates, 0, WindATSPConfig(max_endurance_seconds=1000.0),
    )


def fake_fields(turbine_path, config_path, field_count, seed, field_offset=0, **kwargs):
    records = kwargs.get("generation_records")
    if records is not None:
        records.extend({"field_id": field_offset + index, "seed": seed + index, "accepted": True}
                       for index in range(field_count))
    return [small_instance(f"virtual-{field_offset + index:05d}-000", float(seed + index))
            for index in range(field_count)]


class TrainingPoolTests(unittest.TestCase):
    def make_pool(self) -> TrainingPool:
        initial = [small_instance("real-depot", 0.0), *fake_fields(None, None, 2, 1234)]
        validation = [small_instance("virtual-1000000-000", 1001234.0)]
        return TrainingPool(initial, validation, ROOT / "turbine102.tsv", ROOT / "stage1/config.json",
                            seed=1234, refresh_every_epochs=2, refresh_fields=1)

    @patch("stage2.pool.build_kde_instances", side_effect=fake_fields)
    def test_refresh_replaces_whole_fields_but_preserves_real_regions(self, builder) -> None:
        pool = self.make_pool()
        before = [instance_fingerprint(item) for item in pool.validation_instances]
        self.assertFalse(pool.maybe_refresh(0))
        self.assertFalse(pool.maybe_refresh(1))
        self.assertTrue(pool.maybe_refresh(2))
        self.assertFalse(pool.maybe_refresh(2))
        self.assertEqual(pool.version, 1)
        names = [item.identifiers[0] for item in pool.instances]
        self.assertEqual(names, ["real-depot", "virtual-00001-000", "virtual-00002-000"])
        self.assertEqual(before, [instance_fingerprint(item) for item in pool.validation_instances])
        self.assertEqual(builder.call_args.args[3], 1236)

    @patch("stage2.pool.build_kde_instances", side_effect=fake_fields)
    def test_checkpoint_restores_the_same_next_refresh(self, builder) -> None:
        uninterrupted = self.make_pool()
        uninterrupted.maybe_refresh(2)
        resumed = self.make_pool()
        resumed.load_state_dict(uninterrupted.state_dict())
        uninterrupted.maybe_refresh(4)
        resumed.maybe_refresh(4)
        self.assertEqual(uninterrupted.state_dict(), resumed.state_dict())

    def test_restore_rejects_changed_refresh_settings(self) -> None:
        source = self.make_pool()
        changed = self.make_pool()
        changed.refresh_every_epochs = 3
        with self.assertRaisesRegex(ValueError, "settings"):
            changed.load_state_dict(source.state_dict())

    def test_training_validation_overlap_is_rejected(self) -> None:
        item = small_instance("real-depot", 0.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            TrainingPool([item], [item], ROOT / "turbine102.tsv", ROOT / "stage1/config.json")

    def test_stratified_batches_are_unique_reproducible_and_cover_sizes(self) -> None:
        instances = [small_instance(f"virtual-{index:05d}-000", float(index), nodes=3)
                     for index in range(20)]
        instances += [small_instance("real-big", 100.0, nodes=22)]
        first = sample_training_indices(instances, 4, np.random.default_rng(12), stratified=True)
        second = sample_training_indices(instances, 4, np.random.default_rng(12), stratified=True)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(set(first.tolist())), 4)
        self.assertIn(20, first.tolist())

    def test_uniform_sampler_preserves_existing_without_replacement_behavior(self) -> None:
        instances = [small_instance(str(index), float(index)) for index in range(6)]
        expected = np.random.default_rng(2).choice(6, size=4, replace=False)
        actual = sample_training_indices(instances, 4, np.random.default_rng(2), stratified=False)
        np.testing.assert_array_equal(actual, expected)

    def test_serialized_regions_preserve_the_complete_environment(self) -> None:
        import json

        item = small_instance("real-depot", 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            path.write_text(json.dumps({"regions": [instance_to_record(item)]}), encoding="utf-8")
            restored = load_region_instances(path, ROOT / "stage1/config.json")
        self.assertEqual(restored[0].config, item.config)
        self.assertEqual(instance_fingerprint(restored[0]), instance_fingerprint(item))

    def test_source_hashes_ignore_only_windows_linux_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = root / "source.tsv", root / "config.json"
            source.write_bytes(b"x\ty\r\n1\t2\r\n")
            config.write_bytes(b"{\r\n}\r\n")
            first = source_hashes(source, config)
            source.write_bytes(b"x\ty\n1\t2\n")
            config.write_bytes(b"{\n}\n")
            self.assertEqual(first, source_hashes(source, config))
            source.write_bytes(b"x\ty\n1\t3\n")
            self.assertNotEqual(first, source_hashes(source, config))

    @patch("stage2.pool.build_kde_instances", side_effect=fake_fields)
    def test_refresh_cannot_enter_validation_seed_namespace(self, builder) -> None:
        pool = self.make_pool()
        pool.next_field_id = 1_000_000
        with self.assertRaisesRegex(ValueError, "validation seed stream"):
            pool.maybe_refresh(2)
        builder.assert_not_called()

    @patch("stage2.run.build_training_validation_instances")
    def test_prepared_data_is_reused_and_tampering_is_rejected(self, builder) -> None:
        train = [small_instance("real-depot", 0.0), small_instance("virtual-00000-000", 1234.0)]
        validation = [small_instance("virtual-1000000-000", 1001234.0)]
        builder.return_value = (train, validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            arguments = (path, ROOT / "turbine102.tsv", ROOT / "stage1/config.json", 1234, 1, 1, 12.0, 20)
            first = _prepare_or_load_dataset(*arguments)
            before = (path / "validation_regions.json").read_bytes()
            second = _prepare_or_load_dataset(*arguments)
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(first[2], second[2])
            self.assertEqual(before, (path / "validation_regions.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "settings"):
                _prepare_or_load_dataset(path, arguments[1], arguments[2], 1235, 1, 1, 12.0, 20)
            (path / "validation_regions.json").write_bytes(before + b" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                _prepare_or_load_dataset(*arguments)


if __name__ == "__main__":
    unittest.main()
