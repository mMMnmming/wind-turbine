"""CLI for reproducible Stage-2 preparation, training, search, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
import shutil
import time

from .data import (
    KDE_RETRY_SEED_STRIDE,
    build_deployment_instances,
    build_training_validation_instances,
    instance_to_record,
    instance_fingerprint,
    load_region_instances,
    split_fingerprint,
)
from .pool import TrainingPool, assert_disjoint, source_hashes


ROOT = Path(__file__).resolve().parent


def _resolve_input_path(mode: str, input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    default_filename = "turbine87.tsv" if mode == "evaluate" else "turbine102.tsv"
    return ROOT.parent / default_filename


def _instances_as_dict(instances: list[object]) -> list[dict[str, object]]:
    return [
        {**instance_to_record(item),
            "wind_speed_mps": item.config.wind_speed_mps,
            "gust_speed_mps": item.config.gust_speed_mps,
            "wind_direction_degrees": item.config.wind_direction_degrees,
        }
        for item in instances
    ]


def _write_split_manifest(output: Path, name: str, instances: list[object]) -> None:
    (output / f"{name}_regions.json").write_text(
        json.dumps({"split": name, "regions": _instances_as_dict(instances)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_data_manifest(
    output: Path,
    input_path: Path,
    stage1_config_path: Path,
    seed: int,
    virtual_fields: int,
    validation_fields: int,
    gust_speed_mps: float,
    max_attempts_per_field: int = 20,
    generation_records: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, str]:
    input_hashes = source_hashes(input_path, stage1_config_path)
    manifest = {
        "train_source": str(input_path),
        "validation_source_distribution": str(input_path),
        "seed": seed,
        "training_kde_fields": virtual_fields,
        "validation_kde_fields": validation_fields,
        "validation_seed": seed + 1_000_000,
        "gust_speed_mps": gust_speed_mps,
        "input_hashes": input_hashes,
        "source_hash_policy": "SHA256 after CRLF-to-LF normalization; prepared JSON files use raw-byte hashes.",
        "kde_generation": {
            "policy": "Deterministic rejection sampling conditional on Stage1 deployment feasibility.",
            "max_attempts_per_field": max_attempts_per_field,
            "retry_seed_stride": KDE_RETRY_SEED_STRIDE,
            "attempts_by_split": generation_records or {},
        },
        "note": "Validation KDE fields use a disjoint seed stream; test input is evaluated separately.",
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return input_hashes


def _prepare_or_load_dataset(
    directory: Path, input_path: Path, config_path: Path, seed: int,
    virtual_fields: int, validation_fields: int, gust_speed_mps: float, max_attempts: int,
) -> tuple[list[object], list[object], dict[str, object]]:
    """Prepare once or verify a portable immutable dataset before reuse."""
    settings = {
        "seed": seed, "virtual_fields": virtual_fields, "validation_fields": validation_fields,
        "gust_speed_mps": gust_speed_mps, "max_attempts": max_attempts,
        "input_hashes": source_hashes(input_path, config_path),
    }
    manifest_path = directory / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2 or manifest.get("settings") != settings:
            raise ValueError("Prepared-data settings differ or use an old schema. Use a new --data-dir.")
        for filename, expected_hash in manifest["file_hashes"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Prepared-data hash mismatch: {filename}.")
        train = load_region_instances(directory / "training_regions.json", config_path)
        validation = load_region_instances(directory / "validation_regions.json", config_path)
        assert_disjoint(train, validation)
        return train, validation, manifest
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("training_regions.json", "validation_regions.json")):
        raise ValueError("Incomplete prepared data exists. Use a new --data-dir rather than overwrite it.")
    if not 1 <= virtual_fields < 1_000_000 or validation_fields < 1:
        raise ValueError("Positive field counts are required; training fields must stay below 1000000.")
    records: dict[str, list[dict[str, object]]] = {}
    train, validation = build_training_validation_instances(
        input_path, config_path, virtual_fields, seed, validation_fields, gust_speed_mps,
        max_attempts_per_field=max_attempts, generation_records=records,
    )
    assert_disjoint(train, validation)
    _write_split_manifest(directory, "training", train)
    _write_split_manifest(directory, "validation", validation)
    from .evaluation import nearest_neighbor_route

    baseline_rows = []
    for instance in validation:
        metrics = instance.evaluate_route(nearest_neighbor_route(instance))
        baseline_rows.append({
            "node_count": instance.node_count, "route": list(metrics.route),
            "flight_seconds": metrics.flight_seconds, "feasible": metrics.feasible,
        })
    (directory / "validation_baselines.json").write_text(
        json.dumps({"validation_fingerprint": split_fingerprint(validation), "nearest_neighbor": baseline_rows}, indent=2)
        + "\n", encoding="utf-8",
    )
    _write_data_manifest(directory, input_path, config_path, seed, virtual_fields, validation_fields,
                         gust_speed_mps, max_attempts, records)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "schema_version": 2, "settings": settings,
        "training_fingerprint": split_fingerprint(train), "validation_fingerprint": split_fingerprint(validation),
        "file_hashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                        for name in ("training_regions.json", "validation_regions.json", "validation_baselines.json")},
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return train, validation, manifest


def _search_candidates() -> list[tuple[float, int, float]]:
    return list(itertools.product((3e-5, 1e-4, 3e-4), (32, 64), (0.25, 0.5)))


def _validation_from_checkpoint(checkpoint: Path) -> dict[str, float]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {key: float(value) for key, value in payload["best_validation"].items()}


def _rank_key(validation: dict[str, float]) -> tuple[float, float]:
    return validation["feasible_rate"], -validation["mean_normalized_flight_cost"]


def _limit_virtual_fields(instances: list[object], field_count: int) -> list[object]:
    """Keep all real regions plus virtual fields below a requested field count."""
    selected = []
    for instance in instances:
        nest_id = instance.identifiers[instance.depot_index]
        if not nest_id.startswith("virtual-"):
            selected.append(instance)
            continue
        field_id = int(nest_id.split("-")[1])
        if field_id < field_count:
            selected.append(instance)
    return selected


def _run_search(
    train_instances: list[object],
    validation_instances: list[object],
    base_config: object,
    output: Path,
) -> Path:
    from dataclasses import replace

    from .train import train

    search_root = output / "search"
    screening: list[dict[str, object]] = []
    for trial, (learning_rate, virtual_fields, energy_weight) in enumerate(_search_candidates(), start=1):
        trial_config = replace(base_config, learning_rate=learning_rate, energy_weight=energy_weight)
        # Data counts are prepared by the caller for a common maximum. Each
        # candidate filters on deterministic KDE field identifiers.
        trial_instances = _limit_virtual_fields(train_instances, virtual_fields)
        trial_output = search_root / "screen" / f"trial_{trial:02d}"
        trial_output.mkdir(parents=True, exist_ok=True)
        (trial_output / "search_config.json").write_text(
            json.dumps({**asdict(trial_config), "virtual_fields": virtual_fields}, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = train(trial_instances, trial_config, trial_output, validation_instances)
        validation = _validation_from_checkpoint(checkpoint)
        screening.append({
            "trial": trial,
            "learning_rate": learning_rate,
            "virtual_fields": virtual_fields,
            "energy_weight": energy_weight,
            "seed": trial_config.seed,
            "checkpoint": str(checkpoint),
            "validation": validation,
        })
        (search_root / "screening_results.json").write_text(json.dumps(screening, indent=2) + "\n", encoding="utf-8")

    finalists = sorted(screening, key=lambda item: _rank_key(item["validation"]), reverse=True)[:3]
    repeated: list[dict[str, object]] = []
    for finalist_rank, finalist in enumerate(finalists, start=1):
        for seed_offset in range(3):
            seed = int(base_config.seed) + seed_offset
            run_config = replace(
                base_config,
                seed=seed,
                learning_rate=float(finalist["learning_rate"]),
                energy_weight=float(finalist["energy_weight"]),
            )
            virtual_fields = int(finalist["virtual_fields"])
            trial_instances = _limit_virtual_fields(train_instances, virtual_fields)
            run_output = search_root / "replicates" / f"candidate_{finalist_rank:02d}" / f"seed_{seed}"
            checkpoint = train(trial_instances, run_config, run_output, validation_instances)
            validation = _validation_from_checkpoint(checkpoint)
            repeated.append({
                "candidate_rank": finalist_rank,
                "learning_rate": run_config.learning_rate,
                "virtual_fields": virtual_fields,
                "energy_weight": run_config.energy_weight,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "validation": validation,
            })

    aggregate: list[dict[str, object]] = []
    for finalist_rank, finalist in enumerate(finalists, start=1):
        runs = [item for item in repeated if item["candidate_rank"] == finalist_rank]
        mean_feasible = sum(item["validation"]["feasible_rate"] for item in runs) / len(runs)
        mean_flight = sum(item["validation"]["mean_normalized_flight_cost"] for item in runs) / len(runs)
        aggregate.append({
            "candidate_rank": finalist_rank,
            "learning_rate": finalist["learning_rate"],
            "virtual_fields": finalist["virtual_fields"],
            "energy_weight": finalist["energy_weight"],
            "mean_feasible_rate": mean_feasible,
            "mean_normalized_flight_cost": mean_flight,
            "runs": runs,
        })
    selected = max(aggregate, key=lambda item: (item["mean_feasible_rate"], -item["mean_normalized_flight_cost"]))
    chosen_run = max(
        [item for item in repeated if item["candidate_rank"] == selected["candidate_rank"]],
        key=lambda item: _rank_key(item["validation"]),
    )
    final_dir = search_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = final_dir / "best.pt"
    shutil.copy2(chosen_run["checkpoint"], final_checkpoint)
    report = {
        "search_space": {
            "learning_rate": [3e-5, 1e-4, 3e-4],
            "virtual_fields": [32, 64],
            "energy_weight": [0.25, 0.5],
            "distance_weight": 1.0,
            "completion_bonus": 5.0,
            "safety_penalty": 20.0,
        },
        "selection_rule": "mean feasible rate, then mean normalized flight cost across 3 seeds",
        "screening": screening,
        "top_three_replicates": repeated,
        "candidate_summary": aggregate,
        "selected_candidate": selected,
        "selected_checkpoint": str(final_checkpoint),
    }
    (search_root / "search_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return final_checkpoint


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train or evaluate the wind-aware pointer network.")
    parser.add_argument("mode", choices=("prepare_data", "train", "resume", "search", "evaluate"))
    parser.add_argument("--input", type=Path, help="Defaults to turbine102.tsv for training and turbine87.tsv for evaluation.")
    parser.add_argument("--stage1-config", type=Path, default=ROOT.parent / "stage1" / "config.json")
    parser.add_argument("--regions-json", type=Path, help="Existing Stage-1 region JSON, for evaluation.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Immutable prepared training/validation data; defaults to --output.")
    parser.add_argument("--seed", type=int, help="Policy training seed, independent of --data-seed.")
    parser.add_argument("--data-seed", type=int, help="Seed for prepared data and refreshed KDE fields.")
    parser.add_argument("--refresh-every-epochs", type=int, help="Refresh before epoch interval+1; zero disables it.")
    parser.add_argument("--refresh-fields", type=int)
    parser.add_argument("--stratified-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reference-results", type=Path, help="Optional offline Gurobi reference_results.json.")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--virtual-fields", type=int)
    parser.add_argument("--validation-fields", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--energy-weight", type=float)
    parser.add_argument("--gust-speed-mps", type=float)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--evaluation-seed", type=int, default=1234)
    parser.add_argument("--plot-dpi", type=int, default=300)
    args = parser.parse_args()
    args.input = _resolve_input_path(args.mode, args.input)

    saved_config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("device", "seed", "data_seed", "epochs", "batch_size", "batches_per_epoch", "virtual_fields",
                 "learning_rate", "energy_weight", "gust_speed_mps", "refresh_every_epochs", "refresh_fields",
                 "stratified_sampling"):
        value = getattr(args, name)
        if value is not None:
            saved_config[name] = value
    device = str(saved_config["device"])
    validation_fields = (
        args.validation_fields if args.validation_fields is not None
        else int(saved_config.get("validation_fields", 16))
    )
    gust_speed_mps = float(saved_config.get("gust_speed_mps", 12.0))
    max_kde_attempts = int(saved_config.get("kde_max_attempts_per_field", 20))

    if args.mode == "evaluate":
        instances = load_region_instances(args.regions_json, args.stage1_config, gust_speed_mps) if args.regions_json else build_deployment_instances(args.input, args.stage1_config, gust_speed_mps)
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint PATH")
        if args.random_samples < 1:
            parser.error("--random-samples must be positive")
        return _evaluate(args, parser, instances, device)

    if args.regions_json is not None:
        parser.error("--regions-json is evaluation-only; train and search from turbine102.tsv.")
    if validation_fields < 1:
        parser.error("--validation-fields must be positive")
    if max_kde_attempts < 1:
        parser.error("kde_max_attempts_per_field must be positive")
    if args.mode == "resume" and args.checkpoint is None:
        parser.error("resume requires --checkpoint PATH")
    if args.mode == "search" and args.reference_results is not None:
        parser.error("The legacy search does not consume references; use train or evaluate with --reference-results.")
    if args.mode == "search" and (args.output / "search").exists() and any((args.output / "search").iterdir()):
        parser.error("Search output already contains results. Use a new --output; existing trial files were not overwritten.")
    virtual_fields = int(saved_config.get("virtual_fields", 64))
    refresh_every = int(saved_config.get("refresh_every_epochs", 0))
    refresh_fields = int(saved_config.get("refresh_fields", 0))
    if args.mode in ("train", "resume"):
        if refresh_every < 0 or refresh_fields < 0:
            parser.error("Refresh interval and field count must be nonnegative.")
        if refresh_every and not 1 <= refresh_fields <= virtual_fields:
            parser.error("--refresh-fields must be between 1 and --virtual-fields; or disable refresh with --refresh-every-epochs 0.")
        if args.mode == "train" and any((args.output / name).exists() for name in ("latest.pt", "best.pt")):
            parser.error("Output contains checkpoints. Use a new --output or explicitly resume.")
    max_fields = 64 if args.mode == "search" else virtual_fields
    data_seed = int(saved_config.get("data_seed", saved_config["seed"]))
    data_directory = args.data_dir or args.output
    train_instances, validation_instances, data_manifest = _prepare_or_load_dataset(
        data_directory, args.input, args.stage1_config, data_seed, max_fields,
        validation_fields, gust_speed_mps, max_kde_attempts,
    )
    if args.mode == "search":
        virtual_fields = max_fields
    args.output.mkdir(parents=True, exist_ok=True)
    input_hashes = data_manifest["input_hashes"]
    generation_records = data_manifest["kde_generation"]["attempts_by_split"]
    rejected_draws = sum(
        not record["accepted"] for records in generation_records.values() for record in records
    )
    print(
        f"Prepared {len(train_instances)} training and {len(validation_instances)} validation regions "
        f"from {virtual_fields}+{validation_fields} KDE fields; rejected draws={rejected_draws}.", flush=True,
    )
    if args.mode == "prepare_data":
        return 0

    from .train import TrainingConfig, train

    config_fields = {field.name for field in fields(TrainingConfig)}
    training_options = {key: value for key, value in saved_config.items() if key in config_fields}
    training_config = TrainingConfig(**training_options)
    if args.mode == "search":
        print("Search retains the legacy 32/64-field fixed-data grid; refreshing and stratification are disabled.", flush=True)
        from dataclasses import replace
        training_config = replace(training_config, stratified_sampling=False)
        checkpoint = _run_search(train_instances, validation_instances, training_config, args.output)
    else:
        selected_train_instances = _limit_virtual_fields(train_instances, virtual_fields)
        pool = TrainingPool(
            selected_train_instances, validation_instances, args.input, args.stage1_config, seed=data_seed,
            refresh_every_epochs=refresh_every, refresh_fields=refresh_fields, max_attempts_per_field=max_kde_attempts,
        )
        from .benchmark import load_references
        references = load_references(args.reference_results, validation_instances) if args.reference_results else None
        checkpoint = train(
            selected_train_instances,
            training_config,
            args.output,
            validation_instances,
            resume_checkpoint=args.checkpoint if args.mode == "resume" else None,
            pool=pool, references=references,
        )
    (args.output / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation-selected checkpoint: {checkpoint}")
    return 0


def _timed_greedy_route(model: object, features: object) -> tuple[tuple[int, ...], float]:
    """Time one existing greedy decode and host transfer, not feature creation."""
    import torch

    if features.device.type == "cuda":
        torch.cuda.synchronize(features.device)
    started = time.perf_counter()
    with torch.no_grad():
        order, _ = model(features, decode_type="greedy")
        route = (0, *(int(index) for index in order[0].cpu().tolist()), 0)
    if features.device.type == "cuda":
        torch.cuda.synchronize(features.device)
    return route, time.perf_counter() - started


def _resolve_evaluation_environment(
    instances: list[object], trained_environment: dict, *, frozen_input: bool,
) -> list[object]:
    """Explicit region inputs are authoritative, never silently rewritten."""
    from dataclasses import replace
    from .atsp import WindATSPConfig

    if not trained_environment:
        if frozen_input:
            logging.getLogger(__name__).warning(
                "Checkpoint lacks wind environment metadata; evaluating the explicit frozen input unchanged.")
        return instances
    environment = WindATSPConfig(**trained_environment)
    if frozen_input:
        if any(asdict(instance.config) != asdict(environment) for instance in instances):
            raise ValueError("Checkpoint environment differs from frozen regions. Input wind/endurance was not overridden; use a matching checkpoint or a separately defined experiment.")
        return instances
    return [replace(instance, config=environment) for instance in instances]


def _model_totals(records: list[dict], expected_count: int) -> dict:
    """Report all measured tours, explicitly including infeasible model routes."""
    feasible = sum(bool(record["feasible"]) for record in records)
    return {
        "region_count": expected_count, "reported_count": len(records), "route_region_count": len(records),
        "total_flight_seconds": math.fsum(record["flight_seconds"] for record in records) if len(records) == expected_count else None,
        "available_flight_seconds": math.fsum(record["flight_seconds"] for record in records),
        "total_runtime_seconds": math.fsum(record["runtime_seconds"] for record in records),
        "feasible_region_count": feasible, "infeasible_region_count": len(records) - feasible,
        "all_routes_feasible": feasible == expected_count,
        "flight_total_complete": len(records) == expected_count,
        "flight_total_note": "All measured model tours, including unsafe routes; not the valid-gap subset.",
    }


def _evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser, instances: list[object], device: str) -> int:
    import torch

    from stage1.deployment import load_config
    from .evaluation import evaluate_route, nearest_neighbor_route, random_route_summary, write_evaluation_artifacts
    from .model import GraphPointerNetwork, make_node_features

    source_name = (args.regions_json or args.input).stem
    evaluation_output = args.evaluation_output or args.output / "evaluations" / f"{source_name}_{args.checkpoint.stem}"
    if any((evaluation_output / name).exists() for name in ("data_manifest.json", "reference_results.json")):
        parser.error("Evaluation output must be separate from immutable training/reference data.")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload["training_config"]
    trained_environment = payload.get("wind_environment", {})
    instances = _resolve_evaluation_environment(instances, trained_environment, frozen_input=args.regions_json is not None)
    from .benchmark import load_references, reference_summary
    references = load_references(args.reference_results, instances) if args.reference_results else None
    model = GraphPointerNetwork(config["hidden_dim"], config["encoder_layers"], config["heads"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    evaluations = []
    model_metrics = []
    for region, instance in enumerate(instances, start=1):
        coordinates = torch.tensor(instance.coordinates_km, dtype=torch.float32, device=device).unsqueeze(0)
        features = make_node_features(
            coordinates,
            instance.config.wind_speed_mps,
            instance.config.wind_direction_degrees,
            instance.config.gust_speed_mps,
        )
        if instance.node_count == 1:
            route, inference_seconds = (0, 0), 0.0
        else:
            route, inference_seconds = _timed_greedy_route(model, features)
        model_report = evaluate_route(instance, "model_greedy", route)
        model_metrics.append({
            "region": region, "nest_id": instance.identifiers[instance.depot_index],
            "node_count": instance.node_count, "instance_fingerprint": instance_fingerprint(instance),
            "instance": instance_to_record(instance), "route": list(route),
            "flight_seconds": model_report.flight_seconds, "runtime_seconds": inference_seconds,
            "equivalent_energy_seconds": model_report.equivalent_energy_seconds, "feasible": model_report.feasible,
        })
        nearest_report = evaluate_route(instance, "nearest_neighbor", nearest_neighbor_route(instance))
        random_report = random_route_summary(instance, args.random_samples, args.evaluation_seed + region - 1)
        evaluations.append((region, instance, model_report, nearest_report, random_report))
        print(
            f"Region {region:02d} | nest={instance.identifiers[0]} | nodes={instance.node_count} | "
            f"model flight={model_report.flight_seconds:.2f}s | inference={inference_seconds:.6f}s | "
            f"feasible={model_report.feasible} | nearest flight={nearest_report.flight_seconds:.2f}s | "
            f"random feasible={random_report.feasible_count}/{random_report.sample_count}", flush=True,
        )
    input_hashes = {
        "test_input_sha256": hashlib.sha256((args.regions_json or args.input).read_bytes()).hexdigest(),
        "stage1_config_sha256": hashlib.sha256(args.stage1_config.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    service_radius = load_config(args.stage1_config).communication_radius_km
    if args.regions_json:
        source_payload = json.loads(args.regions_json.read_text(encoding="utf-8"))
        if isinstance(source_payload, dict) and source_payload.get("frozen_reference"):
            service_radius = float(source_payload["source"]["communication_radius_km"])
            if not math.isfinite(service_radius) or service_radius <= 0:
                raise ValueError("Frozen service radius must be finite and positive.")
    write_evaluation_artifacts(
        output=evaluation_output,
        evaluations=evaluations,
        checkpoint_path=args.checkpoint,
        input_hashes=input_hashes,
        service_radius_km=service_radius,
        plot_dpi=args.plot_dpi,
    )
    summary_path = evaluation_output / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training_config"] = config
    if references is not None:
        summary["reference_comparison"] = reference_summary(
            [(instance, report.flight_seconds, report.feasible) for _, instance, report, _, _ in evaluations],
            references,
        )
        summary["reference_results_sha256"] = hashlib.sha256(args.reference_results.read_bytes()).hexdigest()
    summary["training_wind_environment"] = trained_environment
    summary["checkpoint_selection_rule"] = payload.get("selection_rule", "legacy checkpoint; selection rule unavailable")
    summary["test_selection_note"] = "Test metrics are reporting-only and are not used for checkpoint or hyperparameter selection."
    split_name = "test"
    if args.regions_json:
        source_payload = json.loads(args.regions_json.read_text(encoding="utf-8"))
        if isinstance(source_payload, dict) and source_payload.get("split") in ("training", "validation"):
            split_name = source_payload["split"]
    summary["evaluation_split"] = split_name
    totals = _model_totals(model_metrics, len(instances))
    summary["model_totals"] = totals
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (evaluation_output / f"{split_name}_regions.json").write_text(
        json.dumps({"split": split_name, "regions": _instances_as_dict(instances)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path = evaluation_output / "model_metrics.json"
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "schema_version": 1, "method": "model_greedy", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": input_hashes["checkpoint_sha256"], "device": str(device),
        "evaluation_split": split_name,
        "summary": totals, "split_fingerprint": split_fingerprint(instances),
        "timing_scope": "One greedy forward/decode and route transfer to host, synchronized on CUDA; excludes model loading, features, baselines, plotting and file I/O; no warm-up or repeats.",
        "results": model_metrics,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    print(f"Model totals | regions={len(instances)} | flight={totals['total_flight_seconds']:.6f}s | "
          f"inference={totals['total_runtime_seconds']:.6f}s | infeasible={totals['infeasible_region_count']}", flush=True)
    print(f"Evaluated {len(instances)} routing regions. Results: {evaluation_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
