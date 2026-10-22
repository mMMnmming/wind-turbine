"""Auditable multi-objective program evolution; never consults test scores."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .artifacts import RunArtifacts, run_path, safe_child, redact_secrets
from .candidate import validate_candidate_source
from .client import DeepSeekClient, DeepSeekSettings
from .data import fingerprint, prepare_data, turbines_from_record
from .domain import Turbine, result_as_dict, effective_service_radius_km
from .evaluator import evaluate_proposal
from .exports import write_front
from .reward import OBJECTIVE_VERSION, pareto_front, parent_pool, search_reward
from .sandbox import run_candidate
from .selection import DiscountedUCB, SlidingWindowUCB
from .settings import SearchSettings

SEARCH_VERSION = 4
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
INITIAL_DIRECTIONS = ("structural_exploration", "rebalance_assignments", "reduce_nests")


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


def _diagnostic_metrics(record, keys):
    # Legacy failures may contain real margins; drop sentinels, not negatives.
    return {
        k: value
        for k in keys
        if isinstance(value := record.get("metrics", {}).get(k), (int, float))
        and math.isfinite(value)
        and abs(value) < 1e8
    }


class SearchController:
    def __init__(
        self,
        settings: SearchSettings,
        llm: DeepSeekSettings,
        outputs: Path,
        client: DeepSeekClient | None = None,
    ):
        self.settings, self.llm, self.outputs = settings, llm, outputs
        self.client = client if client is not None else DeepSeekClient(llm)
        self.mock = not isinstance(self.client, DeepSeekClient)
        self.archive: list[dict] = []
        self.repair_pool: list[dict] = []
        self.history: list[dict] = []
        self.source_hashes: set[str] = set()
        self.behavior_hashes: set[str] = set()
        self.bandit = (
            SlidingWindowUCB(
                list(settings.direction_actions),
                settings.ucb_window,
                settings.ucb_exploration,
                settings.seed,
            )
            if settings.selector == "sliding_window"
            else DiscountedUCB(
                list(settings.direction_actions),
                settings.ucb_decay,
                settings.ucb_exploration,
                settings.seed,
            )
        )
        self.rng = random.Random(settings.seed)
        self.stalled = 0
        self.pending: dict | None = None
        self.elapsed_seconds = 0.0

    def _identity(self, bundle: dict, system: str) -> dict:
        config = asdict(self.settings)
        for key in (
            "evaluation_budget",
            "heartbeat_seconds",
            "show_code",
            "data_dir",
            "target_input",
            "validation_inputs",
            "protected_input",
        ):
            config.pop(key)
        implementation = {
            p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        }
        return _jsonable(
            {
                "search_version": SEARCH_VERSION,
                "settings": config,
                "data_fingerprint": bundle["fingerprint"],
                "objective_version": OBJECTIVE_VERSION,
                "prompt_hash": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                "implementation": implementation,
                "origin": "offline_mock" if self.mock else "llm",
                "llm": {k: v for k, v in asdict(self.llm).items() if k != "api_key"},
            }
        )

    def _persist(self, artifacts: RunArtifacts) -> None:
        artifacts.write_json(
            "search_state.json",
            {
                "search_version": SEARCH_VERSION,
                "identity": self.identity,
                "evaluation_budget": self.settings.evaluation_budget,
                "archive": self.archive,
                "repair_pool": self.repair_pool,
                "history": self.history,
                "source_hashes": sorted(self.source_hashes),
                "behavior_hashes": sorted(self.behavior_hashes),
                "ucb": self.bandit.state_dict(),
                "rng_state": self.rng.getstate(),
                "stalled": self.stalled,
                "pending": self.pending,
                "active_search_seconds": self.elapsed_seconds
                + time.perf_counter()
                - self.active_start,
            },
        )

    def _restore(self, artifacts: RunArtifacts) -> None:
        state = json.loads(
            (artifacts.root / "search_state.json").read_text(encoding="utf-8")
        )
        if state.get("search_version") != SEARCH_VERSION:
            raise ValueError(
                "Legacy search version cannot resume v4; use a new run ID."
            )
        if state.get("identity") != self.identity:
            raise ValueError(
                "Run identity/configuration changed; do not mix old data, constraints or objectives."
            )
        if self.settings.evaluation_budget < state["evaluation_budget"]:
            raise ValueError("Resume may only increase the evaluation budget.")
        self.archive, self.repair_pool, self.history = (
            state["archive"],
            state["repair_pool"],
            state["history"],
        )
        self.source_hashes, self.behavior_hashes = set(state["source_hashes"]), set(
            state["behavior_hashes"]
        )
        self.bandit.load_state_dict(state["ucb"])

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value

        self.rng.setstate(tuples(state["rng_state"]))
        self.stalled, self.pending = state["stalled"], state["pending"]
        self.elapsed_seconds = state["active_search_seconds"]
        if self.pending and self.pending.get("stage") == "source_ready":
            source = self.pending["algorithm_code"]
            if (
                hashlib.sha256(source.encode("utf-8")).hexdigest()
                != self.pending["source_sha256"]
            ):
                raise ValueError("Pending source hash changed in checkpoint.")
            artifacts.write_text(self.pending["source_path"], source)
            self.pending.pop("algorithm_code")
            self.pending["stage"] = "evaluation"
            artifacts.log(
                f"{self.pending['candidate_id']} recovered saved LLM response without new API request",
                event="source_recovered",
            )
            self._log_candidate(artifacts, self.pending, source)
        # Detect edits to archived or pending source before any new API request.
        for row in [*self.history, *([self.pending] if self.pending else [])]:
            if row.get("source_path") and row.get("source_sha256"):
                text = safe_child(artifacts.root, row["source_path"]).read_text(
                    encoding="utf-8"
                )
                if (
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    != row["source_sha256"]
                ):
                    raise ValueError("Saved candidate source hash changed.")

    def _choose(self) -> tuple[dict | None, str, dict, str]:
        independent_count = sum(
            row.get("selection_reason") == "independent_initial" for row in self.history
        )
        # An unusable source should receive its own traceback on the very next
        # round, rather than paying for more unrelated cold starts first.
        urgent_repair = bool(self.repair_pool) and not self.archive
        if independent_count < self.settings.initial_candidates and not urgent_repair:
            direction = INITIAL_DIRECTIONS[independent_count % 3]
            return None, direction, {}, "independent_initial"
        evolved_count = len(self.history) - independent_count
        repair_due = not self.archive or (evolved_count + 1) % 4 == 0
        if self.repair_pool and repair_due:
            parent = self.repair_pool[-1]
            text = " ".join(parent["evaluation"].get("violations", [])).lower()
            diagnostic_patterns = [
                ("coverage", "coverage"),
                ("self-assignment", "coverage"),
                ("undeclared nest", "undeclared_nest"),
                ("refer to an input", "invalid_nest"),
                ("endurance", "endurance"),
                ("service-radius", "service_radius"),
            ]
            direction = next(
                (
                    self.settings.diagnostic_direction_map[kind]
                    for needle, kind in diagnostic_patterns
                    if needle in text and kind in self.settings.diagnostic_direction_map
                ),
                "constraint_repair",
            )
            direction, trace = self.bandit.select_with_trace(direction)
            return parent, direction, trace, "repair_failed_child"
        pool = parent_pool(self.archive, self.settings.archive_size)
        parent = self.rng.choice(pool) if pool else None
        if self.settings.selector == "sliding_window":
            eligible = [
                a
                for a in self.settings.direction_actions
                if a != "constraint_repair" or parent is None
            ]
            direction, trace = self.bandit.select_with_trace(eligible=eligible)
        else:
            direction, trace = self.bandit.select_with_trace()
        return (
            parent,
            direction,
            trace,
            "diverse_pareto_parent" if parent else "no_source_restart",
        )

    def _log_candidate(self, artifacts: RunArtifacts, row: dict, source: str) -> None:
        candidate_id = row["candidate_id"]
        for key in (
            "algorithm_idea",
            "change_hypothesis",
            "expected_improvement",
            "change_summary",
        ):
            artifacts.log(
                f"{candidate_id} {key}: {row.get(key,'N/A')}", event="llm_explanation"
            )
        if self.settings.show_code:
            numbered = "\n".join(
                f"{index:04d} | {line}"
                for index, line in enumerate(source.splitlines(), 1)
            )
            artifacts.log(
                f"{candidate_id} returned source BEGIN (not executed/validated yet)\n{numbered}\n"
                f'{candidate_id} returned source END; file={row["source_path"]}',
                event="llm_code",
            )
        else:
            artifacts.log(
                f'{candidate_id} source display disabled; file={row["source_path"]}',
                event="llm_code",
            )

    def _observe_response(
        self, artifacts: RunArtifacts, message: str, fields: dict
    ) -> None:
        candidate_id = self.pending["candidate_id"] if self.pending else "unknown"
        kind = fields.pop("kind", "api_request")
        if kind == "llm_text":
            if self.settings.show_code:
                artifacts.log(
                    f"{candidate_id} {fields['text']}",
                    event="llm_text",
                    candidate_id=candidate_id,
                )
        elif kind == "llm_response":
            path = f"responses/{candidate_id}.json"
            artifacts.write_json(path, fields)
            self.pending["response_path"] = path
            artifacts.log(
                f"{candidate_id} finish_reason={fields.get('finish_reason')} "
                f"response_chars={fields.get('response_chars')} usage={fields.get('usage')} "
                f"public_response={path}",
                event="llm_response",
                candidate_id=candidate_id,
                response_path=path,
                finish_reason=fields.get("finish_reason"),
                usage=fields.get("usage"),
            )
            if not self.llm.stream and self.settings.show_code:
                artifacts.log(
                    f"{candidate_id} PUBLIC RESPONSE (non-streaming; not yet validated)\n"
                    f"{fields.get('content') or '<empty final content>'}",
                    event="llm_text",
                )
        elif kind == "llm_response_invalid":
            path = self.pending.get("response_path")
            if not self.settings.show_code:
                artifacts.log(
                    f"{candidate_id} {message}; public body display disabled; diagnostic={path}",
                    event=kind,
                    candidate_id=candidate_id,
                )
                return
            content = getattr(self.client, "last_response", {}).get("content", "")
            # Adapters may expose only last_metadata, so read the already saved
            # redacted public body instead of ever dumping a provider object.
            if path:
                saved = json.loads(
                    safe_child(artifacts.root, path).read_text(encoding="utf-8")
                )
                content = saved.get("content", "")
            visible = content[:12000] or "<empty final content>"
            suffix = (
                f"\n[console limited to 12000 chars; complete public body: {path}]"
                if len(content) > 12000
                else ""
            )
            artifacts.log(
                f"{candidate_id} {message}\nPUBLIC RESPONSE BEGIN\n{visible}{suffix}\n"
                f"PUBLIC RESPONSE END; diagnostic={path}",
                event=kind,
                candidate_id=candidate_id,
            )
        else:
            artifacts.log(
                f"{candidate_id} {message}",
                event=kind,
                candidate_id=candidate_id,
                **fields,
            )

    def _prompt(
        self, parent: dict | None, direction: str, artifacts: RunArtifacts
    ) -> str:
        feedback = {
            "direction_id": direction,
            "deployment_config": asdict(self.settings.deployment),
            "objective_definition": OBJECTIVE_VERSION,
            "dataset_count": len(self.bundle["datasets"]),
            "parent_diagnostic": parent["evaluation"] if parent else None,
        }
        # No protected coordinates, test results, API credentials or raw whole data.
        if parent:
            # Never send complete partitions/proposals or repeat aggregate violations.
            # Keep every failed instance's numeric diagnostics and the whole source.
            metric_keys = (
                "nest_count",
                "total_proxy_flight_seconds",
                "maximum_region_tasks",
                "minimum_endurance_margin_seconds",
                "minimum_service_margin_km",
            )
            feedback["parent_diagnostic"] = {
                "status": parent["evaluation"].get("status"),
                "objectives": parent.get("objectives"),
                "datasets": [
                    {
                        "name": r["name"],
                        "feasible": r["feasible"],
                        "metrics": _diagnostic_metrics(r, metric_keys),
                        "violations": list(dict.fromkeys(r.get("violations", [])))[:4],
                        "diagnostics": r.get("diagnostics", [])[:4],
                    }
                    for r in sorted(
                        parent["evaluation"].get("datasets", []),
                        key=lambda d: d["feasible"],
                    )[: self.settings.feedback_dataset_limit]
                ],
                "aggregate": parent["evaluation"].get("aggregate"),
                "diagnostics": parent["evaluation"].get("diagnostics", [])[:4],
            }
            if (
                not parent["evaluation"].get("datasets")
                or parent["evaluation"].get("status") == "candidate_error"
            ):
                feedback["parent_diagnostic"]["violations"] = parent["evaluation"].get(
                    "violations", []
                )
            for key in ("failure_stage", "failed_dataset"):
                if parent["evaluation"].get(key):
                    feedback["parent_diagnostic"][key] = parent["evaluation"][key]
        feedback["recent_attempts"] = [
            {
                "direction": r["direction_id"],
                "decision": r["decision"],
                "reward": r["reward"],
                "delta": r.get("objective_delta"),
            }
            for r in self.history[-3:]
        ]
        feedback["work_limits"] = self._limits()
        text = json.dumps(
            feedback, ensure_ascii=False, separators=(",", ":"), default=str
        )
        source = (
            (artifacts.root / parent["source_path"]).read_text(encoding="utf-8")
            if parent
            else "无；从零生成。"
        )
        return f"本轮方向必须为 {direction}。\n固定评测诊断：{text}\n父代完整源码：\n{source}"

    def _limits(self) -> dict:
        return {
            "soft_seconds": min(
                self.settings.soft_seconds,
                self.settings.candidate_timeout_seconds * 0.8,
            ),
            "max_region_evaluations": self.settings.max_region_evaluations,
            "max_move_attempts": self.settings.max_move_attempts,
        }

    def _constraint_preflight(self, artifacts, row, source_path):
        if not self.settings.constraint_preflight:
            return
        row["evaluation_phase"] = "constraint_preflight"
        done = row.setdefault("constraint_checks", [])
        if any(not d["feasible"] for d in done):
            raise ValueError("Saved constraint preflight failed.")
        radius = effective_service_radius_km(self.settings.deployment)
        cases = [
            ("separated_clusters", [(0.0, 0.0), (radius * 3, 0.0), (radius * 3, 0.01)]),
            (
                "endurance_ring",
                [
                    (0.0, 0.0),
                    *[
                        (
                            radius * 0.98 * math.cos(i * math.tau / 12),
                            radius * 0.98 * math.sin(i * math.tau / 12),
                        )
                        for i in range(12)
                    ],
                ],
            ),
        ]
        for name, xy in cases[len(done) :]:
            points = [
                Turbine(f"constraint-{i:03d}", x, y) for i, (x, y) in enumerate(xy)
            ]
            artifacts.log(
                f"{row['candidate_id']} hard-constraint preflight={name} start",
                event="constraint_preflight_start",
            )
            started = time.perf_counter()
            with artifacts.heartbeat(
                f"{row['candidate_id']} constraint preflight={name}",
                self.settings.heartbeat_seconds,
            ):
                proposal = run_candidate(
                    source_path,
                    points,
                    self.settings.deployment,
                    self.settings.seed,
                    min(10, self.settings.candidate_timeout_seconds),
                    limits=self._limits(),
                )
                checked = evaluate_proposal(points, self.settings.deployment, proposal)
            check = {
                "name": name,
                "feasible": checked.feasible,
                "violations": list(checked.violations),
                "diagnostics": list(checked.diagnostics),
                "seconds": time.perf_counter() - started,
            }
            done.append(check)
            self._persist(artifacts)
            artifacts.log(
                f"{row['candidate_id']} preflight={name} feasible={checked.feasible} "
                f"diagnostics={check['diagnostics']}",
                event="constraint_preflight_end",
            )
            if not checked.feasible:
                raise ValueError(
                    f"Constraint preflight {name} failed: {'; '.join(checked.violations)}"
                )
        if any(not d["feasible"] for d in done):
            raise ValueError("Saved constraint preflight failed.")

    def _preflight(self, artifacts: RunArtifacts, row: dict, source_path: Path) -> None:
        """Cheap interface/runtime checks, not deployment scores or feasibility proof."""
        row["evaluation_phase"] = "preflight"
        check = row.setdefault(
            "preflight",
            {
                "status": "running",
                "cases": [],
                "origin": "synthetic_interface_check",
                "seconds": 0.0,
            },
        )
        if check["status"] == "passed":
            return
        if check["status"] == "failed":
            raise RuntimeError(check["error"])
        cases = (
            ("single", [(0.0, 0.0)]),
            ("small", [(0.0, 0.0), (0.1, 0.0), (0.03, 0.12), (0.11, 0.09)]),
        )
        for name, coordinates in cases[len(check["cases"]) :]:
            points = [
                Turbine(f"preflight-{i}", x, y) for i, (x, y) in enumerate(coordinates)
            ]
            timeout = min(5, self.settings.candidate_timeout_seconds)
            artifacts.log(
                f"{row['candidate_id']} case={name} nodes={len(points)} timeout={timeout}s; "
                "interface/runtime only, not a deployment score",
                event="preflight_start",
            )
            start = time.perf_counter()
            try:
                with artifacts.heartbeat(
                    f"{row['candidate_id']} preflight={name}",
                    self.settings.heartbeat_seconds,
                ):
                    run_candidate(
                        source_path,
                        points,
                        self.settings.deployment,
                        self.settings.seed,
                        timeout,
                    )
            except Exception as error:
                elapsed = time.perf_counter() - start
                check.update(
                    status="failed",
                    failed_case=name,
                    error=f"{type(error).__name__}: {error}",
                )
                check["seconds"] += elapsed
                self._persist(artifacts)
                artifacts.log(
                    f"{row['candidate_id']} case={name} FAIL after {elapsed:.3f}s: {error}",
                    event="preflight_end",
                )
                raise
            elapsed = time.perf_counter() - start
            check["cases"].append(
                {"name": name, "nodes": len(points), "seconds": elapsed}
            )
            check["seconds"] += elapsed
            self._persist(artifacts)
            artifacts.log(
                f"{row['candidate_id']} case={name} PASS in {elapsed:.3f}s; "
                "full deployment checks still required",
                event="preflight_end",
            )
        check["status"] = "passed"
        self._persist(artifacts)

    def _evaluate(self, artifacts: RunArtifacts, row: dict) -> dict:
        row["evaluation_phase"] = "source_validation"
        source_path = safe_child(artifacts.root, row["source_path"])
        source = source_path.read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != row["source_sha256"]:
            raise ValueError("Candidate source changed before evaluation.")
        validate_candidate_source(source)
        if row["source_sha256"] in self.source_hashes:
            return {
                "status": "duplicate_source",
                "feasible": False,
                "duplicate": True,
                "violations": [],
            }
        self._preflight(artifacts, row, source_path)
        self._constraint_preflight(artifacts, row, source_path)
        datasets = row.setdefault("completed_datasets", [])
        records = (
            []
            if self.settings.early_rejection
            and any(not d["feasible"] for d in datasets)
            else self.bundle["datasets"][len(datasets) :]
        )
        for record in records:
            row["evaluation_phase"] = "dataset"
            row["active_dataset"] = record["name"]
            turbines = turbines_from_record(record)
            artifacts.log(
                f"{row['candidate_id']} dataset={record['name']} nodes={len(turbines)} start",
                event="dataset_start",
            )
            start = time.perf_counter()
            telemetry = {}
            try:
                with artifacts.heartbeat(
                    f"{row['candidate_id']} dataset={record['name']} running",
                    self.settings.heartbeat_seconds,
                ):
                    proposal = run_candidate(
                        source_path,
                        turbines,
                        self.settings.deployment,
                        self.settings.seed,
                        self.settings.candidate_timeout_seconds,
                        limits=self._limits(),
                        telemetry=telemetry,
                        observer=lambda progress: artifacts.log(
                            f"{row['candidate_id']} dataset={record['name']} candidate-reported progress={progress}",
                            event="algorithm_progress",
                            progress=progress,
                        ),
                    )
            except (Exception, KeyboardInterrupt):
                row.setdefault("failed_attempts", []).append(
                    {
                        "dataset": record["name"],
                        "algorithm_seconds": time.perf_counter() - start,
                        "telemetry": telemetry,
                    }
                )
                raise
            algorithm_seconds = time.perf_counter() - start
            check_start = time.perf_counter()
            with artifacts.heartbeat(
                f"{row['candidate_id']} dataset={record['name']} evaluator",
                self.settings.heartbeat_seconds,
            ):
                checked = evaluate_proposal(
                    turbines, self.settings.deployment, proposal
                )
            checked_seconds = time.perf_counter() - check_start
            data = {
                "name": record["name"],
                "split": record["split"],
                "turbine_count": len(turbines),
                "feasible": checked.feasible,
                "metrics": asdict(checked.metrics),
                "violations": list(checked.violations),
                "algorithm_seconds": algorithm_seconds,
                "diagnostics": list(checked.diagnostics),
                "telemetry": telemetry,
                "evaluation_seconds": checked_seconds,
                "proposal": asdict(proposal),
                "recomputed_deployment": (
                    result_as_dict(checked.deployment) if checked.deployment else None
                ),
            }
            datasets.append(data)
            self._persist(artifacts)
            artifacts.log(
                f"{row['candidate_id']} dataset={record['name']} feasible={checked.feasible} "
                f"nests={checked.metrics.nest_count if checked.feasible else 'N/A'} "
                f"proxy_flight={checked.metrics.total_proxy_flight_seconds:.3f}s "
                f"max_tasks={checked.metrics.maximum_region_tasks} "
                f"endurance_margin={checked.metrics.minimum_endurance_margin_seconds:.3f}s "
                f"radius_margin={checked.metrics.minimum_service_margin_km:.4f}km "
                f"algorithm={algorithm_seconds:.3f}s evaluator={checked_seconds:.3f}s "
                f"violations={'; '.join(checked.violations) or 'none'} work={telemetry}",
                event="dataset_end",
            )
            if not checked.feasible and self.settings.early_rejection:
                break
        behavior = fingerprint(
            [
                {
                    "nest_ids": sorted(d["proposal"]["nest_ids"]),
                    "assignments": d["proposal"]["assignments"],
                }
                for d in datasets
            ]
        )
        feasible = len(datasets) == len(self.bundle["datasets"]) and all(
            d["feasible"] for d in datasets
        )
        duplicate = feasible and behavior in self.behavior_hashes
        result = {
            "status": "feasible" if feasible else "infeasible",
            "feasible": feasible,
            "datasets": datasets,
            "target": datasets[0],
            "behavior_sha256": behavior,
            "duplicate": duplicate,
            "violations": [
                f"{d['name']}: {v}" for d in datasets for v in d["violations"]
            ],
        }
        result.update(
            evaluated_count=len(datasets),
            total_count=len(self.bundle["datasets"]),
            complete=len(datasets) == len(self.bundle["datasets"]),
            diagnostics=[
                dict(diag, dataset=d["name"])
                for d in datasets
                for diag in d["diagnostics"]
            ],
        )
        if not result["complete"]:
            result["duplicate"] = (
                False  # partial matching is not behavioral equivalence
            )
            result.pop("behavior_sha256")
        if feasible:
            metrics = [d["metrics"] for d in datasets]
            objectives = [
                sum(d["metrics"]["nest_count"] / d["turbine_count"] for d in datasets)
                / len(datasets),
                sum(
                    d["metrics"]["total_proxy_flight_seconds"] / d["turbine_count"]
                    for d in datasets
                )
                / len(datasets),
                max(
                    d["metrics"]["maximum_region_tasks"] / d["turbine_count"]
                    for d in datasets
                ),
            ]
            result["aggregate"] = {
                "mean_nest_count": sum(m["nest_count"] for m in metrics) / len(metrics),
                "mean_total_proxy_flight_seconds": sum(
                    m["total_proxy_flight_seconds"] for m in metrics
                )
                / len(metrics),
                "maximum_region_tasks": max(m["maximum_region_tasks"] for m in metrics),
            }
            result["objectives"] = objectives
        return result

    def _finish(self, artifacts: RunArtifacts, row: dict, evaluation: dict) -> None:
        before = {tuple(r["objectives"]) for r in self.archive}
        evaluation["source_sha256"] = row.get("source_sha256")
        row["evaluation"] = evaluation
        row["objectives"] = evaluation.get("objectives")
        parent = next(
            (r for r in self.history if r["candidate_id"] == row["parent_id"]), None
        )
        duplicate = bool(evaluation.get("duplicate"))
        if evaluation["feasible"] and not duplicate:
            self.archive = pareto_front([*self.archive, row])
        after = {tuple(r["objectives"]) for r in self.archive}
        new_front = before != after
        row["reward"] = search_reward(
            evaluation["feasible"],
            row["objectives"],
            parent.get("objectives") if parent else None,
            duplicate=duplicate,
            new_front=new_front,
            violation_count=len(evaluation.get("violations", [])),
            violation_severity=(
                sum(d.get("normalized_excess", 0.1) for d in evaluation["diagnostics"])
                if evaluation.get("diagnostics")
                else None
            ),
        )
        if evaluation["feasible"] and not before and not duplicate:
            row["reward"] = 1.0
        if evaluation.get("status") == "generation_error":
            row["bandit_credit"] = "excluded_generation_error"
        elif (
            self.settings.selector == "sliding_window"
            and row["selection_reason"] != "diverse_pareto_parent"
        ):
            row["bandit_credit"] = "separate_initial_or_repair"
        else:
            self.bandit.update(row["direction_id"], row["reward"])
            row["bandit_credit"] = "optimization"
        if not evaluation["feasible"] and not duplicate and row.get("source_path"):
            self.repair_pool = [*self.repair_pool, row][-self.settings.archive_size :]
        row["decision"] = (
            "duplicate"
            if duplicate
            else (
                "front_member"
                if row in self.archive
                else "dominated" if evaluation["feasible"] else "infeasible_or_error"
            )
        )
        if parent and row["objectives"] and parent.get("objectives"):
            row["objective_delta"] = [
                a - b for a, b in zip(row["objectives"], parent["objectives"])
            ]
        if row.get("source_sha256"):
            self.source_hashes.add(row["source_sha256"])
        if evaluation.get("behavior_sha256"):
            self.behavior_hashes.add(evaluation["behavior_sha256"])
        row.pop("completed_datasets", None)
        self.history.append(row)
        self.pending = None
        self.stalled = 0 if new_front else self.stalled + 1
        self._persist(artifacts)
        self._reports(artifacts)
        artifacts.log(
            f"{row['candidate_id']} reward={row['reward']:+.2f} decision={row['decision']} "
            f"objectives={row['objectives']} delta={row.get('objective_delta')} "
            f"front={len(self.archive)} stalled={self.stalled}/{self.settings.no_improvement_limit}",
            event="iteration_end",
            candidate_id=row["candidate_id"],
            reward=row["reward"],
            decision=row["decision"],
        )

    def _reports(self, artifacts: RunArtifacts) -> None:
        write_front(artifacts.root, self.archive)
        groups = {}
        for row in self.history:
            ev = row["evaluation"]
            if ev.get("feasible") and ev.get("behavior_sha256"):
                groups.setdefault(ev["behavior_sha256"], []).append(row)
        artifacts.write_json(
            "efficiency_alternatives.json",
            {
                "note": "Same frozen-manifest behavior only; timings are single-run observations, not statistical speedups.",
                "groups": [
                    {
                        "behavior_sha256": key,
                        "candidates": [
                            {
                                "candidate_id": r["candidate_id"],
                                "algorithm_seconds": sum(
                                    d["algorithm_seconds"]
                                    for d in r["evaluation"].get("datasets", [])
                                ),
                            }
                            for r in rows
                        ],
                    }
                    for key, rows in groups.items()
                ],
            },
        )
        for row in self.history:
            artifacts.write_json(f"evaluations/{row['index']:04d}.json", row)
        with (artifacts.root / "trajectory.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "candidate",
                    "parent",
                    "direction",
                    "feasible",
                    "reward",
                    "decision",
                    "objectives",
                    "llm_seconds",
                    "algorithm_seconds",
                    "evaluation_seconds",
                    "preflight_seconds",
                ]
            )
            for row in self.history:
                data = row["evaluation"].get("datasets", [])
                writer.writerow(
                    [
                        row["candidate_id"],
                        row["parent_id"],
                        row["direction_id"],
                        row["evaluation"]["feasible"],
                        row["reward"],
                        row["decision"],
                        row["objectives"],
                        row.get("llm_seconds", 0),
                        sum(d["algorithm_seconds"] for d in data)
                        + sum(
                            a["algorithm_seconds"]
                            for a in row.get("failed_attempts", [])
                        ),
                        sum(d["evaluation_seconds"] for d in data),
                        row.get("preflight", {}).get("seconds", 0.0),
                    ]
                )

    def run(
        self, run_id: str | None = None, resume: bool = False, reset_stall: bool = False
    ) -> Path:
        self.bundle = prepare_data(self.settings)
        system = (PROMPTS / "system.md").read_text(encoding="utf-8")
        self.identity = self._identity(self.bundle, system)
        if not resume and run_id and run_path(self.outputs, run_id).exists():
            raise FileExistsError(
                "Run directory exists; choose a new run ID or resume."
            )
        if resume and (
            not run_id
            or not (run_path(self.outputs, run_id) / "search_state.json").exists()
        ):
            raise FileNotFoundError("No resumable search state.")
        artifacts = RunArtifacts(self.outputs, run_id, secrets=(self.llm.api_key,))
        self.active_start = time.perf_counter()
        if resume:
            self._restore(artifacts)
            if reset_stall:
                self.stalled = 0
        artifacts.write_json("config_snapshot.json", self.identity)
        if hasattr(self.client, "set_observer"):

            def observe(message, **fields):
                self._observe_response(artifacts, message, fields)

            self.client.set_observer(observe)
        artifacts.log(
            f"Search v4 origin={'offline_mock' if self.mock else 'llm'} datasets={len(self.bundle['datasets'])} "
            f"completed={len(self.history)} budget={self.settings.evaluation_budget}; objective={OBJECTIVE_VERSION}",
            event="search_start",
        )
        self._persist(artifacts)
        try:
            while (
                len(self.history) < self.settings.evaluation_budget
                and self.stalled < self.settings.no_improvement_limit
            ):
                if self.pending is None:
                    parent, direction, trace, reason = self._choose()
                    index = len(self.history) + 1
                    self.pending = {
                        "index": index,
                        "candidate_id": f"C{index:04d}",
                        "parent_id": parent["candidate_id"] if parent else None,
                        "direction_id": direction,
                        "direction_selection": trace,
                        "selection_reason": reason,
                        "origin": "offline_mock" if self.mock else "llm",
                        "stage": "generation",
                    }
                    self._persist(artifacts)
                    row = self.pending
                    artifacts.log(
                        f"round={index}/{self.settings.evaluation_budget} remaining={self.settings.evaluation_budget-index} "
                        f"parent={row['parent_id']} direction={direction} reason={reason}",
                        event="iteration_start",
                        direction_selection=trace,
                    )
                    start = time.perf_counter()
                    try:
                        artifacts.log(
                            f"{row['candidate_id']} LLM request start",
                            event="llm_start",
                        )
                        with artifacts.heartbeat(
                            f"{row['candidate_id']} LLM request active",
                            self.settings.heartbeat_seconds,
                        ):
                            generated = self.client.generate(
                                system,
                                self._prompt(parent, direction, artifacts),
                                direction,
                            )
                        if (
                            redact_secrets(
                                generated.algorithm_code, (self.llm.api_key,)
                            )
                            != generated.algorithm_code
                        ):
                            raise ValueError(
                                "Generated source contains sensitive-looking text; refusing unsafe checkpoint."
                            )
                    except Exception as error:
                        row["llm_seconds"] = time.perf_counter() - start
                        row["api_metadata"] = getattr(
                            self.client,
                            "last_metadata",
                            {"actual_requests": 0, "note": "offline mock"},
                        )
                        artifacts.log(
                            f"{row['candidate_id']} generation failed: {type(error).__name__}: {error}",
                            event="llm_error",
                        )
                        self._finish(
                            artifacts,
                            row,
                            {
                                "status": "generation_error",
                                "feasible": False,
                                "violations": [f"{type(error).__name__}: {error}"],
                            },
                        )
                        continue
                    row.update(asdict(generated))
                    row["source_path"] = f"candidates/{index:04d}.py"
                    row["source_sha256"] = hashlib.sha256(
                        generated.algorithm_code.encode("utf-8")
                    ).hexdigest()
                    row["llm_seconds"] = time.perf_counter() - start
                    row["api_metadata"] = getattr(
                        self.client,
                        "last_metadata",
                        {"actual_requests": 0, "note": "offline mock"},
                    )
                    row["stage"] = "source_ready"
                    # Save returned source before writing its standalone file.
                    # A killed writer can recover without paying for regeneration.
                    self._persist(artifacts)
                    artifacts.write_text(row["source_path"], generated.algorithm_code)
                    row.pop("algorithm_code")
                    row["stage"] = "evaluation"
                    self._log_candidate(artifacts, row, generated.algorithm_code)
                    artifacts.log(
                        f"{row['candidate_id']} LLM finished in {row['llm_seconds']:.3f}s",
                        event="llm_end",
                        api_metadata=row["api_metadata"],
                    )
                    self._persist(artifacts)
                row = self.pending
                if row["stage"] == "generation":
                    # Request may have reached the provider; do not silently repeat it.
                    self._finish(
                        artifacts,
                        row,
                        {
                            "status": "generation_error",
                            "feasible": False,
                            "violations": [
                                "Interrupted generation: API outcome unknown; round consumed without replay."
                            ],
                        },
                    )
                    continue
                try:
                    evaluation = self._evaluate(artifacts, row)
                except Exception as error:
                    evaluation = {
                        "status": "candidate_error",
                        "feasible": False,
                        "datasets": row.get("completed_datasets", []),
                        "violations": [f"{type(error).__name__}: {error}"],
                        "failure_stage": row.get("evaluation_phase", "unknown"),
                        "failed_dataset": row.get("active_dataset"),
                        "diagnostics": [
                            d
                            for c in row.get("constraint_checks", [])
                            for d in c["diagnostics"]
                        ],
                    }
                    artifacts.log(
                        f"{row['candidate_id']} phase={evaluation['failure_stage']} "
                        f"dataset={evaluation['failed_dataset']} candidate error: {error}",
                        event="candidate_error",
                    )
                self._finish(artifacts, row, evaluation)
        except (KeyboardInterrupt, MemoryError, OSError):
            self._persist(artifacts)
            artifacts.log(
                "Interrupted; saved completed work and pending round. Use resume.",
                event="interrupted",
            )
            raise
        self._reports(artifacts)
        status = "best_within_budget" if self.archive else "no_deployable_llm_candidate"
        reason = (
            "budget_exhausted"
            if len(self.history) >= self.settings.evaluation_budget
            else "front_stalled"
        )
        artifacts.write_json(
            "summary.json",
            {
                "status": status,
                "stop_reason": reason,
                "evaluated_rounds": len(self.history),
                "pareto_candidates": [r["candidate_id"] for r in self.archive],
                "global_optimality_proven": False,
                "origin": "offline_mock" if self.mock else "llm",
                "active_search_seconds": self.elapsed_seconds
                + time.perf_counter()
                - self.active_start,
                "selection": "User must explicitly select a Pareto candidate; no implicit best algorithm.",
                "generated_sources": sum(
                    bool(row.get("source_path")) for row in self.history
                ),
                "preflight_passed": sum(
                    row.get("preflight", {}).get("status") == "passed"
                    for row in self.history
                ),
                "fully_evaluated_candidates": sum(
                    len(row["evaluation"].get("datasets", []))
                    == len(self.bundle["datasets"])
                    for row in self.history
                ),
            },
        )
        self._persist(artifacts)
        if not self.archive and self.repair_pool and reason == "budget_exhausted":
            artifacts.log(
                "No deployable candidate; failed source and diagnostics saved. Generation budget exhausted; "
                "no extra API call was made. Resume this NEW-version run with a larger budget to allow repair.",
                event="repair_budget_exhausted",
            )
        artifacts.log(
            f"Search complete status={status} stop={reason} front={len(self.archive)} output={artifacts.root}",
            event="search_end",
        )
        return artifacts.root
