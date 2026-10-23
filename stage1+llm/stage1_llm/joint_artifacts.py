"""V5 rankings, versioned selected runtimes and actual network route exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import atomic_json, atomic_text, safe_child

RUNTIME_FILES = {
    "__init__.py",
    "candidate_tools.py",
    "domain.py",
    "routing.py",
    "candidate.py",
    "sandbox.py",
    "joint_tools.py",
    "joint_planner.py",
    "joint_moves.py",
    "joint_evidence.py",
}


def read_state(root: Path) -> dict:
    state = json.loads((root / "search_state.json").read_text(encoding="utf-8"))
    if state.get("search_version") != 5:
        raise ValueError("Expected v5 result; legacy searches retain legacy semantics.")
    return state


def list_joint_candidates(root: Path):
    rows = read_state(root)["archive"]
    print(
        "候选 | 平均机巢数 | 平均全场总飞行秒 | 平均区域飞行秒 | 最坏区域飞行秒",
        flush=True,
    )
    for row in rows:
        m = row["evaluation"]["aggregate"]
        print(
            f"{row['candidate_id']} | {m['mean_nest_count']:.6f} | {m['mean_total_flight_seconds']:.6f} | "
            f"{m['mean_region_flight_seconds']:.6f} | {m['maximum_flight_seconds']:.6f}",
            flush=True,
        )
    return rows


def verify_selection(candidate: Path, metadata: dict):
    """Verify selected code/runtime/model before use, never silently replace helpers."""
    selection = candidate.parent / "selection.json"
    if not selection.is_file():
        return
    record = json.loads(selection.read_text(encoding="utf-8"))
    if record.get("search_version") != 5:
        raise ValueError(
            "Legacy selected algorithm cannot be silently evaluated with v5 tools."
        )
    if record["oracle"] != metadata:
        raise ValueError(
            "Selected algorithm requires its original Stage2 checkpoint/environment/device/implementation."
        )
    if (
        hashlib.sha256(candidate.read_text(encoding="utf-8").encode()).hexdigest()
        != record["source_sha256"]
    ):
        raise ValueError("Selected source hash changed.")
    required = (
        RUNTIME_FILES
        if record.get("tool_contract_version", 4) >= 5
        else RUNTIME_FILES - {"joint_moves.py", "joint_evidence.py"}
    )
    if set(record["runtime_hashes"]) != required:
        raise ValueError("Incomplete selected runtime manifest.")
    for name, digest in record["runtime_hashes"].items():
        if (
            hashlib.sha256(
                (candidate.parent / "runtime/stage1_llm" / name).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise ValueError("Selected runtime hash changed.")


def export_joint_candidate(root: Path, candidate_id=None, *, allow_mock=False):
    state = read_state(root)
    eligible = [
        r
        for r in [
            *state["history"],
            *([state["seed_candidate"]] if state.get("seed_candidate") else []),
        ]
        if r["evaluation"]["feasible"] and r["evaluation"].get("complete")
    ]
    if candidate_id is None:
        if not state["archive"]:
            raise ValueError("No deployable candidate to export.")
        candidate_id = state["archive"][0]["candidate_id"]
    row = next((r for r in eligible if r["candidate_id"] == candidate_id), None)
    if row is None:
        raise ValueError("Export requires a complete feasible candidate.")
    if row["origin"] == "offline_mock" and not allow_mock:
        raise ValueError("Offline mock cannot be exported as a formal LLM algorithm.")
    source = safe_child(root, row["source_path"]).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != row["source_sha256"]:
        raise ValueError("Candidate hash changed.")
    contents = {}
    for name in RUNTIME_FILES:
        content = (Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")
        if (
            name not in state["identity"]["implementation"]
            or hashlib.sha256(content).hexdigest()
            != state["identity"]["implementation"][name]
        ):
            raise ValueError(
                "Trusted source changed after search; export original implementation instead."
            )
        contents[name] = content
    output = safe_child(root, f"selected/{candidate_id}")
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(output / "algorithm.py", source)
    runtime = output / "runtime/stage1_llm"
    hashes = {}
    for name, content in contents.items():
        atomic_text(runtime / name, content.decode("utf-8"))
        hashes[name] = hashlib.sha256((runtime / name).read_bytes()).hexdigest()
    atomic_json(
        output / "selection.json",
        {
            "search_version": 5,
            "tool_contract_version": state["tool_contract_version"],
            "candidate_id": candidate_id,
            "origin": row["origin"],
            "source_sha256": row["source_sha256"],
            "runtime_hashes": hashes,
            "oracle": state["identity"]["stage2"],
            "run_identity": state["identity"],
            "selection_rule": "All instances feasible, then mean nests, mean total flight, worst region flight; no test87 feedback.",
        },
    )
    saved_config = json.loads(
        (Path(__file__).resolve().parents[1] / "config/search-v5.json").read_text(
            encoding="utf-8"
        )
    )
    saved_config.update(state["identity"]["settings"])
    # Exported evaluation is independent of the search's seed/schedule.
    saved_config.update(search_mode="standard", seed_candidate=None)
    saved_config.update(
        stage2_device=state["identity"]["stage2"].get("device", "cpu"),
        gust_speed_mps=state["identity"]["stage2"]["environment"]["gust_speed_mps"],
    )
    atomic_json(output / "search-config.json", saved_config)
    atomic_text(
        output / "requirements.txt",
        "numpy>=1.26\nscipy>=1.11\nmatplotlib>=3.8\n# Install matching CUDA PyTorch separately; existing Stage2 repository required.\n",
    )
    atomic_text(
        output / "run_selected.py",
        '''"""Run inside an installed matching wind-turbine repository; model is not bundled."""
import argparse
from pathlib import Path
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--repo-root',type=Path,default=Path.cwd())
args,rest=parser.parse_known_args()
sys.path.insert(0,str(args.repo_root/'stage1+llm'))
from stage1_llm.joint_cli import main
if __name__=='__main__':
    raise SystemExit(main(['evaluate','--search-config',str(Path(__file__).parent/'search-config.json'),'--candidate',str(Path(__file__).parent/'algorithm.py'),*rest]))
''',
    )
    return output


def export_deployment(
    output: Path, points, proposal, checked, metadata, config, *, plots=True
):
    """Only geometry/record/plot work: never run a second policy or baseline."""
    import numpy as np

    from stage2.atsp import WindATSPConfig, WindATSPInstance
    from stage2.data import instance_fingerprint, instance_to_record

    if (output / "summary.json").exists() or (output / "regions.json").exists():
        raise FileExistsError(
            "Evaluation already exported; use a new output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    by_id = {p.identifier: p for p in points}
    records, signatures, nests, metrics = [], [], [], []
    for index, (nest, row) in enumerate(
        zip(sorted(proposal.nest_ids), checked["regions"]), 1
    ):
        ids = [
            nest,
            *sorted(
                t
                for t, owner in proposal.assignments.items()
                if owner == nest and t != nest
            ),
        ]
        instance = WindATSPInstance(
            tuple(ids),
            np.asarray([(by_id[t].x_km, by_id[t].y_km) for t in ids]),
            0,
            WindATSPConfig(**metadata["environment"]),
        )
        signature = instance_fingerprint(instance)
        records.append(instance_to_record(instance))
        signatures.append(signature)
        nests.append(
            {
                "identifier": nest,
                "x_km": by_id[nest].x_km,
                "y_km": by_id[nest].y_km,
                "turbine_ids": ids,
                "route_ids": row.get("route_ids"),
                "task_count": len(ids) - 1,
                "flight_seconds": row.get("flight_seconds"),
                "equivalent_energy_seconds": row.get("energy_seconds"),
                "endurance_margin_seconds": row.get("endurance_margin_seconds"),
                "service_radius_margin_km": row.get("radius_margin_km"),
                "wind_audit": row.get("wind_audit"),
            }
        )
        metrics.append(
            {
                "region": index,
                "nest_id": nest,
                "node_count": len(ids),
                "instance_fingerprint": signature,
                "instance": instance_to_record(instance),
                "route": (
                    [ids.index(t) for t in row["route_ids"]]
                    if row.get("route_ids")
                    else None
                ),
                "route_ids": row.get("route_ids"),
                "flight_seconds": row.get("flight_seconds"),
                "runtime_seconds": row.get("runtime_seconds"),
                "feasible": row["feasible"],
                "method": "model_greedy",
            }
        )
        if plots and row.get("route_ids"):
            import matplotlib.pyplot as plt

            from stage2.route_plotting import draw_energy_route

            route = tuple(ids.index(t) for t in row["route_ids"])
            fig, ax = plt.subplots(figsize=(7, 6))
            draw_energy_route(ax, instance, route, "Frozen Stage2 greedy")
            ax.set_title(
                f"Region {index} | {'feasible' if row['feasible'] else 'INFEASIBLE'}"
            )
            fig.savefig(
                output / f"region_{index:02d}_network_route.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)
    source = {
        "origin": metadata["origin"],
        "oracle": metadata,
        "route_method": "frozen_stage2_greedy",
        "global_optimality_proven": False,
    }
    atomic_json(
        output / "regions.json",
        {
            "schema_version": 1,
            "split": "test",
            "source": source,
            "regions": records,
            "instance_fingerprints": signatures,
            "split_fingerprint": hashlib.sha256(
                "\n".join(signatures).encode()
            ).hexdigest(),
        },
    )
    atomic_json(
        output / "summary.json",
        {
            "config": asdict(config),
            "nest_count": len(nests),
            "nests": nests,
            "effective_service_radius_km": config.communication_radius_km,
            "assignments": proposal.assignments,
            "all_constraints_satisfied": checked["feasible"],
            "metrics": checked["metrics"],
            "source": source,
            "metric_notes": "Actual directed wind-aware greedy flight; depot hover excluded; no nearest-neighbour or proxy audit.",
        },
    )
    atomic_json(
        output / "model_metrics.json",
        {
            "schema_version": 1,
            "method": "model_greedy",
            **metadata,
            "results": metrics,
            "candidate_count": 1,
            "expected_region_count": len(metrics),
            "totals": checked["metrics"],
            "timing_scope": "Stage2 single greedy inference; excludes model loading, feature construction, plotting and export. Cached timings are original observations, not additional inference time.",
        },
    )
