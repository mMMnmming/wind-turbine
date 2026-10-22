"""Manual selection and read-compatible, complete geometry exports."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

from .artifacts import atomic_json, safe_child
from .domain import Turbine, result_as_dict
from .evaluator import Evaluation


def load_state(root: Path) -> dict:
    state = json.loads((root / "search_state.json").read_text(encoding="utf-8"))
    if state.get("search_version") not in (3, 4):
        raise ValueError("Expected a readable v3/v4 search state.")
    return state


def list_candidates(root: Path) -> list[dict]:
    rows = load_state(root)["archive"]
    print(
        "候选 | 平均机巢数 | 平均总飞行时间/s(启发式) | 最大区域任务数 | 三目标",
        flush=True,
    )
    for row in rows:
        summary = row["evaluation"]["aggregate"]
        print(
            f"{row['candidate_id']} | {summary['mean_nest_count']:.3f} | "
            f"{summary['mean_total_proxy_flight_seconds']:.3f} | {summary['maximum_region_tasks']} | "
            f"{row['objectives']}",
            flush=True,
        )
    return rows


def candidate_origin(candidate: Path, source: str) -> dict:
    """Do not relabel a mock or arbitrary external program as a formal result."""
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    selection = candidate.parent / "selection.json"
    if selection.is_file():
        record = json.loads(selection.read_text(encoding="utf-8"))
        if record["source_sha256"] != actual:
            raise ValueError("Selected algorithm source hash changed.")
        return {
            "origin": record["origin"],
            "selected": True,
            "candidate_id": record["candidate_id"],
        }
    state_path = candidate.parent.parent / "search_state.json"
    if candidate.parent.name == "candidates" and state_path.is_file():
        state = load_state(candidate.parent.parent)
        for row in state["history"]:
            if row.get("source_sha256") == actual:
                return {
                    "origin": row["origin"],
                    "selected": False,
                    "candidate_id": row["candidate_id"],
                }
    return {"origin": "unverified_external", "selected": False}


def export_candidate(
    root: Path,
    candidate_id: str,
    *,
    allow_mock: bool = False,
    reason: str = "User selected from frozen Pareto table.",
) -> Path:
    state = load_state(root)
    selected = next(
        (row for row in state["archive"] if row["candidate_id"] == candidate_id), None
    )
    if selected is None or not selected["evaluation"]["feasible"]:
        raise ValueError(
            "Candidate must be a feasible member of the saved Pareto front."
        )
    if selected["origin"] == "offline_mock" and not allow_mock:
        raise ValueError("Cannot export offline mock as a formal LLM algorithm.")
    source = safe_child(root, selected["source_path"]).read_text(encoding="utf-8")
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        != selected["evaluation"]["source_sha256"]
    ):
        raise ValueError("Candidate source hash changed since evaluation.")
    if state.get("search_version") == 4:
        for name in (
            "__init__.py",
            "candidate_tools.py",
            "domain.py",
            "routing.py",
            "candidate.py",
            "sandbox.py",
        ):
            content = (
                (Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")
            )
            expected = state["identity"]["implementation"].get(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(
                    "Trusted implementation changed after search; cannot silently export a new runtime."
                )
    destination = safe_child(root, f"selected/{candidate_id}")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "algorithm.py").write_text(source, encoding="utf-8")
    runtime_hashes = snapshot_runtime(destination)
    config = state["identity"]["settings"]["deployment"]
    atomic_json(destination / "deployment.json", config)
    (destination / "run_selected.py").write_text(STANDALONE, encoding="utf-8")
    atomic_json(
        destination / "selection.json",
        {
            "candidate_id": candidate_id,
            "reason": reason,
            "origin": selected["origin"],
            "source_sha256": selected["evaluation"]["source_sha256"],
            "objectives": selected["objectives"],
            "run_identity": state["identity"],
            "runtime_hashes": runtime_hashes,
            "reference_note": "No 87 score was read by this selection operation.",
        },
    )
    atomic_json(destination / "evaluation.json", selected["evaluation"])
    return destination


def snapshot_runtime(destination: Path, candidate: Path | None = None) -> dict:
    """Export a self-contained, hash-checked helper bundle (requires NumPy)."""
    from .sandbox import _runtime

    origin = _runtime(candidate) / "stage1_llm" if candidate else Path(__file__).parent
    target = destination / "runtime" / "stage1_llm"
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in (
        "__init__.py",
        "candidate_tools.py",
        "domain.py",
        "routing.py",
        "candidate.py",
        "sandbox.py",
    ):
        content = (origin / name).read_bytes()
        (target / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


STANDALONE = '''"""Run this selected deployment package without the original repository."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parent
selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))
actual = hashlib.sha256((root/'algorithm.py').read_text(encoding='utf-8').encode('utf-8')).hexdigest()
if actual != selection['source_sha256']:
    raise ValueError('Selected algorithm source hash changed.')
expected_files = {'__init__.py','candidate_tools.py','domain.py','routing.py','candidate.py','sandbox.py'}
if set(selection['runtime_hashes']) != expected_files:
    raise ValueError('Incomplete runtime snapshot.')
for name, expected_hash in selection['runtime_hashes'].items():
    if hashlib.sha256((root/'runtime/stage1_llm'/name).read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Exported runtime hash changed.')
sys.path.insert(0,str(root/'runtime'))
from stage1_llm.domain import DeploymentConfig, load_turbines
from stage1_llm.candidate_tools import CandidateTools
from stage1_llm.sandbox import run_candidate
parser = argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
saved = selection['run_identity']['settings']
parser.add_argument('--seed',type=int,default=saved['seed'])
parser.add_argument('--timeout',type=int,default=saved['candidate_timeout_seconds'])
args = parser.parse_args()
config = DeploymentConfig(**json.loads((root/'deployment.json').read_text(encoding='utf-8')))
expected = selection['run_identity']['settings']['deployment']
if asdict(config) != expected:
    raise ValueError('Exported physical configuration changed.')
points = load_turbines(args.input)
telemetry = {}
limits = {'soft_seconds':min(saved.get('soft_seconds',120),args.timeout*.8),
          'max_region_evaluations':saved.get('max_region_evaluations',5000),
          'max_move_attempts':saved.get('max_move_attempts',20000)}
proposal = run_candidate(root/'algorithm.py',points,config,args.seed,args.timeout,telemetry=telemetry,limits=limits)
checked = CandidateTools(points,config).deployment(asdict(proposal))
print(json.dumps(dict(proposal=asdict(proposal),evaluation=checked,telemetry=telemetry,
    origin=selection['origin'],run_config=dict(seed=args.seed,timeout=args.timeout,limits=limits)),ensure_ascii=False,indent=2))
sys.exit(0 if checked['feasible'] else 2)
'''


def export_evaluation(
    output: Path, turbines: list[Turbine], evaluation: Evaluation, source: dict
) -> None:
    if not evaluation.feasible or evaluation.deployment is None:
        raise ValueError(
            "Cannot export an infeasible deployment as a completed region list."
        )
    if (output / "summary.json").exists() or (output / "regions.json").exists():
        raise FileExistsError("Deployment export already exists; use another run ID.")
    output.mkdir(parents=True, exist_ok=True)
    deployment = evaluation.deployment
    by_id = {t.identifier: t for t in turbines}
    environment = {
        key: value
        for key, value in asdict(deployment.config).items()
        if key not in ("communication_radius_km", "coordinate_unit")
    }
    environment["gust_speed_mps"] = 12.0
    regions = []
    signatures = []
    for nest in deployment.nests:
        identifiers = [
            nest.identifier,
            *(key for key in nest.turbine_ids if key != nest.identifier),
        ]
        record = {
            "identifiers": identifiers,
            "coordinates_km": [
                [float(by_id[key].x_km), float(by_id[key].y_km)] for key in identifiers
            ],
            "depot_index": 0,
            "node_count": len(identifiers),
            "config": environment,
        }
        # Exact encoding used by current Stage2's instance_fingerprint.
        value = {key: v for key, v in record.items() if key != "identifiers"}
        signatures.append(
            hashlib.sha256(
                json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
            ).hexdigest()
        )
        regions.append(record)
    atomic_json(
        output / "regions.json",
        {
            "schema_version": 1,
            "split": "test",
            "frozen_reference": True,
            "source": source,
            "regions": regions,
            "instance_fingerprints": signatures,
            "split_fingerprint": hashlib.sha256(
                "\n".join(signatures).encode()
            ).hexdigest(),
            "note": "Fixed 12 m/s gust feature; proxy routes are heuristic, not proven optimal.",
        },
    )
    summary = result_as_dict(deployment)
    summary["source"] = source
    summary["metric_notes"] = {
        "conservative_energy": "17 m/s audit, hover includes nest turbine.",
        "proxy_energy": "Directed wind times; hover excludes depot. No optimality proof.",
    }
    atomic_json(output / "summary.json", summary)


def write_front(root: Path, rows: list[dict]) -> None:
    atomic_json(
        root / "pareto.json",
        {"solutions": rows, "optimality": "Not proven globally optimal."},
    )
    # Tables are rebuilt from the atomic search state after resume.
    with (root / "pareto.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "候选",
                "平均机巢数",
                "平均总飞行时间/s(启发式)",
                "最大区域任务数",
                "机巢比例",
                "单位风机飞行时间/s",
                "最坏任务比例",
            ]
        )
        for row in rows:
            aggregate = row["evaluation"]["aggregate"]
            writer.writerow(
                [
                    row["candidate_id"],
                    aggregate["mean_nest_count"],
                    aggregate["mean_total_proxy_flight_seconds"],
                    aggregate["maximum_region_tasks"],
                    *row["objectives"],
                ]
            )
