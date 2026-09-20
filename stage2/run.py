"""CLI for preparing data, training, resuming, and evaluating the Stage-2 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import build_deployment_instances, build_kde_instances, load_region_instances


ROOT = Path(__file__).resolve().parent


def _instances_as_dict(instances: list[object]) -> list[dict[str, object]]:
    return [{"identifiers": item.identifiers, "coordinates_km": item.coordinates_km.tolist(), "depot_index": item.depot_index, "node_count": item.node_count} for item in instances]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or evaluate the wind-aware pointer network.")
    parser.add_argument("mode", choices=("prepare_data", "train", "resume", "evaluate"))
    parser.add_argument("--input", type=Path, default=ROOT.parent / "turbine102.tsv")
    parser.add_argument("--stage1-config", type=Path, default=ROOT.parent / "stage1" / "config.json")
    parser.add_argument("--regions-json", type=Path, help="Existing Stage-1 region JSON, for evaluation.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--virtual-fields", type=int)
    args = parser.parse_args()
    saved_config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("device", "epochs", "batch_size", "batches_per_epoch", "virtual_fields"):
        value = getattr(args, name)
        if value is not None:
            saved_config[name] = value
    device = str(saved_config["device"])
    if args.regions_json is not None:
        instances = load_region_instances(args.regions_json, args.stage1_config)
    else:
        instances = build_deployment_instances(args.input, args.stage1_config)
    virtual_fields = int(saved_config.get("virtual_fields", 0))
    if virtual_fields and args.mode in {"prepare_data", "train", "resume"}:
        instances.extend(build_kde_instances(args.input, args.stage1_config, virtual_fields, int(saved_config["seed"])))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_regions.json").write_text(json.dumps(_instances_as_dict(instances), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "input_hashes.json").write_text(json.dumps({
        "turbine_tsv_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "stage1_config_sha256": hashlib.sha256(args.stage1_config.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    if args.mode == "prepare_data":
        print(f"Prepared {len(instances)} Stage-1 routing regions.")
        return 0
    if args.mode in {"train", "resume"}:
        if args.regions_json is not None:
            parser.error("--regions-json is evaluation-only; train from turbine102.tsv.")
        from .train import TrainingConfig, train
        from dataclasses import fields
        if args.mode == "resume" and args.checkpoint is None:
            parser.error("resume requires --checkpoint PATH")
        training_options = {field.name: saved_config[field.name] for field in fields(TrainingConfig)}
        checkpoint = train(
            instances,
            TrainingConfig(**training_options),
            args.output,
            resume_checkpoint=args.checkpoint if args.mode == "resume" else None,
        )
        print(f"Saved checkpoint: {checkpoint}")
        return 0
    if args.checkpoint is None:
        parser.error("evaluate requires --checkpoint PATH")
    from .model import GraphPointerNetwork, make_node_features
    import torch
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload["training_config"]
    model = GraphPointerNetwork(config["hidden_dim"], config["encoder_layers"], config["heads"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    reports = []
    for instance in instances:
        coordinates = torch.tensor(instance.coordinates_km, dtype=torch.float32, device=device).unsqueeze(0)
        features = make_node_features(coordinates, instance.config.wind_speed_mps, instance.config.wind_direction_degrees)
        from .train import directed_time_matrix
        time_matrix = directed_time_matrix(coordinates, instance)
        with torch.no_grad():
            order, _ = model(
                features, decode_type="greedy", travel_time_seconds=time_matrix,
                hover_equivalent_seconds=instance.config.hover_equivalent_seconds,
                max_endurance_seconds=instance.config.max_endurance_seconds,
            )
        route = (0, *(int(index) for index in order[0].cpu().tolist()), 0)
        metrics = instance.evaluate_route(route)
        reports.append({"depot": instance.identifiers[0], "route": [instance.identifiers[index] for index in route], "flight_seconds": metrics.flight_seconds, "equivalent_energy_seconds": metrics.equivalent_energy_seconds, "feasible": metrics.feasible})
    (args.output / "evaluation_routes.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(reports)} routing regions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
