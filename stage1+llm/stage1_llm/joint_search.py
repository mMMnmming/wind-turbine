"""V5 lexicographic program evolution with a frozen wind-aware routing actor."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from .artifacts import RunArtifacts, atomic_text, redact_secrets, run_path, safe_child
from .candidate import CANDIDATE_BUILTINS, validate_candidate_source
from .data import fingerprint, prepare_data, turbines_from_record
from .domain import Turbine
from .joint_planner import InfrastructureError
from .joint_tools import JointTools, aggregate_metrics, ranking_key, tool_reference
from .sandbox import CandidateExecutionError, run_candidate
from .search import SearchController

SEARCH_VERSION = 5
OBJECTIVE = (
    "feasible_all;mean_nests,mean_total_network_flight,max_region_network_flight:lex:v1"
)


@contextmanager
def run_lock(outputs: Path, run_id: str):
    """Exclusive publication; never delete another process's or a stale lock."""
    run_path(outputs, run_id)  # validate before constructing the lock filename
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f".{run_id}.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise FileExistsError(
            f"Run is locked: {path}; verify no process is running before manually clearing a stale lock."
        ) from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


class JointSearch(SearchController):
    """Reuse public-response logging/UCB only, not the legacy physical evaluator."""

    def __init__(self, joint, llm, outputs, planner, client=None):
        super().__init__(joint.search, llm, outputs, client)
        self.joint, self.planner = joint, planner

    def _identity(self, bundle, system):
        identity = super()._identity(bundle, system)
        identity.update(
            search_version=5,
            tool_contract_version=4,
            objective_version=OBJECTIVE,
            stage2=self.planner.metadata,
        )
        return identity

    def _persist(self, artifacts):
        artifacts.write_json(
            "search_state.json",
            {
                "search_version": 5,
                "tool_contract_version": 4,
                "identity": self.identity,
                "evaluation_budget": self.settings.evaluation_budget,
                "archive": self.archive,
                "repair_pool": self.repair_pool,
                "history": self.history,
                "pending": self.pending,
                "stalled": self.stalled,
                "ucb": self.bandit.state_dict(),
                "rng_state": self.rng.getstate(),
                "active_search_seconds": self.elapsed_seconds
                + time.perf_counter()
                - self.active_start,
            },
        )

    def _restore(self, artifacts):
        state = json.loads(
            (artifacts.root / "search_state.json").read_text(encoding="utf-8")
        )
        if state.get("search_version") != 5 or state["identity"] != self.identity:
            raise ValueError(
                "V5 resume identity changed (data, model, environment, device, prompt or implementation); create a new run."
            )
        if self.settings.evaluation_budget < state["evaluation_budget"]:
            raise ValueError("Resume may only increase budget.")
        self.archive, self.repair_pool, self.history = (
            state["archive"],
            state["repair_pool"],
            state["history"],
        )
        self.pending, self.stalled = state["pending"], state["stalled"]
        self.elapsed_seconds = state["active_search_seconds"]
        self.bandit.load_state_dict(state["ucb"])

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value

        self.rng.setstate(tuples(state["rng_state"]))
        for row in [*self.history, *([self.pending] if self.pending else [])]:
            if row.get("source_path"):
                source_path = safe_child(artifacts.root, row["source_path"])
                if row.get("stage") == "source_ready":
                    source = row["algorithm_code"]
                    if fingerprint(source) != row["source_fingerprint"]:
                        raise ValueError("Saved pending source was changed.")
                    artifacts.write_text(row["source_path"], source)
                    row.pop("algorithm_code")
                    row["stage"] = "evaluation"
                source = source_path.read_text(encoding="utf-8")
                if hashlib.sha256(source.encode()).hexdigest() != row["source_sha256"]:
                    raise ValueError("Saved candidate source hash changed.")

    def _choose(self):
        independent = sum(
            r["selection_reason"] == "independent_initial" for r in self.history
        )
        if independent < self.settings.initial_candidates and not (
            self.repair_pool and not self.archive
        ):
            return (
                None,
                ("structural_exploration", "rebalance_assignments", "reduce_nests")[
                    independent % 3
                ],
                {},
                "independent_initial",
            )
        evolved = len(self.history) - independent
        if self.repair_pool and (not self.archive or (evolved + 1) % 4 == 0):
            direction, trace = self.bandit.select_with_trace("constraint_repair")
            return self.repair_pool[-1], direction, trace, "repair_failed_child"
        if self.archive:
            parent = (
                self.archive[0]
                if len(self.archive) == 1 or self.rng.random() < 0.7
                else self.rng.choice(self.archive[1:])
            )
        else:
            parent = None
        eligible = [
            a for a in self.settings.direction_actions if a != "constraint_repair"
        ]
        if self.settings.selector == "sliding_window":
            direction, trace = self.bandit.select_with_trace(eligible=eligible)
        else:
            direction, trace = self.bandit.select_with_trace()
        return (
            parent,
            direction,
            trace,
            "ranked_parent" if parent else "no_source_restart",
        )

    def _prompt(self, parent, direction, artifacts):
        feedback = {
            "direction_id": direction,
            "objective": OBJECTIVE,
            "deployment_config": asdict(self.settings.deployment),
            "environment": self.planner.metadata["environment"],
            "work_limits": self._limits(),
            "dataset_count": len(self.bundle["datasets"]),
        }
        source = "无，从零生成完整算法。"
        if parent:
            evaluation = parent["evaluation"]
            feedback["parent"] = {
                "candidate_id": parent["candidate_id"],
                "objectives": parent.get("objectives"),
                "status": evaluation["status"],
                "violations": evaluation.get("violations", [])[:3],
                "contract_errors": evaluation.get("contract_errors", [])[:3],
                "datasets": [
                    {
                        k: d.get(k)
                        for k in ("name", "split", "feasible", "metrics", "diagnostics")
                    }
                    for d in sorted(
                        evaluation.get("datasets", []),
                        key=lambda d: (
                            d["feasible"],
                            -max(
                                (x.get("excess", 0) for x in d.get("diagnostics", [])),
                                default=0,
                            ),
                        ),
                    )[: self.settings.feedback_dataset_limit]
                ],
            }
            source = safe_child(artifacts.root, parent["source_path"]).read_text(
                encoding="utf-8"
            )
        feedback["recent"] = [
            {
                k: r.get(k)
                for k in (
                    "candidate_id",
                    "direction_id",
                    "decision",
                    "reward",
                    "objective_delta",
                )
            }
            for r in self.history[-3:]
        ]
        return f"固定评测反馈（不是87测试成绩）：{json.dumps(feedback,ensure_ascii=False,allow_nan=False)}\n父代完整源码：\n{source}"

    def _execute(self, artifacts, row, points, label, timeout):
        telemetry = {}
        row["active_check"] = label
        self._persist(artifacts)

        def observe(event):
            if event.get("kind") == "tool_contract_error":
                row.setdefault("observed_contract_errors", []).append(event)
                row["observed_contract_errors"] = row["observed_contract_errors"][:5]
                self._persist(artifacts)
            if event.get("kind") in (
                "network_region",
                "tool_contract_error",
                "repair_start",
                "repair_end",
            ):
                artifacts.log(
                    f"{row['candidate_id']} {label} {event}", event=event["kind"]
                )
            elif event.get("cache_hits", 0):
                artifacts.log(
                    f"{row['candidate_id']} {label} tool progress={event}",
                    event="tool_progress",
                )

        started = time.perf_counter()
        with artifacts.heartbeat(
            f"{row['candidate_id']} {label} deployment/network evaluation active",
            self.settings.heartbeat_seconds,
        ):
            proposal = run_candidate(
                safe_child(artifacts.root, row["source_path"]),
                points,
                self.settings.deployment,
                self.settings.seed,
                timeout,
                limits=self._limits(),
                telemetry=telemetry,
                observer=observe,
                planner=self.planner,
            )
        algorithm_seconds = time.perf_counter() - started
        started = time.perf_counter()
        checked = dict(
            JointTools(
                points,
                self.settings.deployment,
                planner=self.planner,
                soft_seconds=self.settings.candidate_timeout_seconds,
            ).deployment(asdict(proposal))
        )
        return (
            proposal,
            checked,
            telemetry,
            algorithm_seconds,
            time.perf_counter() - started,
        )

    def _evaluate(self, artifacts, row):
        if row.get("observed_contract_errors"):
            errors = row["observed_contract_errors"]
            raise CandidateExecutionError(
                errors[0]["message"],
                telemetry={
                    "contract_errors": errors,
                    "contract_error_count": len(errors),
                },
            )
        row["evaluation_phase"] = "source_validation"
        source = safe_child(artifacts.root, row["source_path"]).read_text(
            encoding="utf-8"
        )
        if hashlib.sha256(source.encode()).hexdigest() != row["source_sha256"]:
            raise ValueError("Candidate source changed before evaluation.")
        validate_candidate_source(source)
        if any(r.get("source_sha256") == row["source_sha256"] for r in self.history):
            return {
                "status": "duplicate_source",
                "feasible": False,
                "duplicate": True,
                "datasets": [],
            }
        row["evaluation_phase"] = "preflight"
        cases = [
            ("single", [Turbine("a", 0, 0)]),
            (
                "small",
                [
                    Turbine("a", 0, 0),
                    Turbine("b", 0.1, 0),
                    Turbine("c", 0, 0.1),
                    Turbine("d", 0.1, 0.1),
                ],
            ),
        ]
        if self.settings.constraint_preflight:
            cases += [
                ("radius", [Turbine("a", 0, 0), Turbine("b", 20, 0)]),
                ("cluster", [Turbine(str(i), i * 0.2, 0) for i in range(5)]),
            ]
        for label, points in cases:
            if label in row.get("completed_preflight", []):
                continue
            artifacts.log(
                f"{row['candidate_id']} preflight {label} start",
                event="preflight_start",
            )
            _, checked, _, _, _ = self._execute(
                artifacts,
                row,
                points,
                label,
                min(5, self.settings.candidate_timeout_seconds),
            )
            if not checked["feasible"]:
                raise ValueError(
                    f"Preflight {label} returned infeasible network deployment: {checked['diagnostics']}"
                )
            row.setdefault("completed_preflight", []).append(label)
            self._persist(artifacts)
            artifacts.log(
                f"{row['candidate_id']} preflight {label} PASS", event="preflight_end"
            )
        datasets = row.setdefault("completed_datasets", [])
        for record in self.bundle["datasets"][len(datasets) :]:
            row["evaluation_phase"] = "dataset"
            artifacts.log(
                f"{row['candidate_id']} dataset={record['name']} deployment start",
                event="dataset_start",
            )
            proposal, checked, telemetry, algorithm_s, evaluation_s = self._execute(
                artifacts,
                row,
                turbines_from_record(record),
                record["name"],
                self.settings.candidate_timeout_seconds,
            )
            entry = {
                "name": record["name"],
                "split": record["split"],
                "turbine_count": len(record["turbines"]),
                "proposal": asdict(proposal),
                **checked,
                "telemetry": telemetry,
                "algorithm_seconds": algorithm_s,
                "evaluation_seconds": evaluation_s,
            }
            datasets.append(entry)
            self._persist(artifacts)
            artifacts.log(
                f"{row['candidate_id']} dataset={record['name']} feasible={checked['feasible']} "
                f"metrics={checked['metrics']} violations={checked['diagnostics']} caches={self.planner.stats()}",
                event="dataset_end",
            )
            if not checked["feasible"] and self.settings.early_rejection:
                break
        complete = len(datasets) == len(self.bundle["datasets"])
        feasible = complete and all(d["feasible"] for d in datasets)
        behavior = fingerprint([d["proposal"] for d in datasets]) if complete else None
        result = {
            "status": "feasible" if feasible else "network_deployment_infeasible",
            "feasible": feasible,
            "datasets": datasets,
            "complete": complete,
            "behavior_sha256": behavior,
            "duplicate": bool(
                behavior
                and any(
                    r["evaluation"].get("behavior_sha256") == behavior
                    for r in self.history
                )
            ),
            "violations": [
                f"{d['name']}: {diag}" for d in datasets for diag in d["diagnostics"]
            ],
        }
        if feasible:
            aggregate = aggregate_metrics(datasets)
            result.update(
                aggregate=aggregate,
                objectives=[
                    aggregate["mean_nest_count"],
                    aggregate["mean_total_flight_seconds"],
                    aggregate["maximum_flight_seconds"],
                ],
            )
        return result

    def _finish(self, artifacts, row, evaluation):
        before = tuple(self.archive[0]["objectives"]) if self.archive else None
        parent = next(
            (r for r in self.history if r["candidate_id"] == row["parent_id"]), None
        )
        row.update(
            evaluation=evaluation,
            objectives=evaluation.get("objectives"),
            origin="offline_mock" if self.mock else "llm",
        )
        feasible, duplicate = evaluation["feasible"], evaluation.get("duplicate", False)
        parent_key = (
            tuple(parent["objectives"]) if parent and parent.get("objectives") else None
        )
        if duplicate:
            reward, decision = 0.0, "duplicate"
        elif not feasible:
            reward, decision = (
                -1.0 - min(1.0, 0.1 * len(evaluation.get("violations", []))),
                "infeasible_or_error",
            )
        elif parent_key is None:
            reward, decision = 1.0, "first_feasible_or_repair"
        else:
            key = ranking_key(row["objectives"])
            reward = 0.75 if key < parent_key else (-0.5 if key > parent_key else 0.0)
            decision = (
                "improved_parent"
                if key < parent_key
                else ("worse_parent" if key > parent_key else "equal_parent")
            )
        if feasible and not duplicate:
            self.archive = sorted(
                [*self.archive, row],
                key=lambda r: (ranking_key(r["objectives"]), r["index"]),
            )[:6]
        if not feasible and row.get("source_path") and not duplicate:
            self.repair_pool = [*self.repair_pool, row][-6:]
        row.update(reward=reward, decision=decision)
        if parent_key and row.get("objectives"):
            row["objective_delta"] = [
                a - b for a, b in zip(row["objectives"], parent_key)
            ]
        if (
            row["selection_reason"] != "independent_initial"
            and evaluation["status"] != "generation_error"
        ):
            self.bandit.update(row["direction_id"], reward)
        after = tuple(self.archive[0]["objectives"]) if self.archive else None
        improved = after is not None and (before is None or after < before)
        self.stalled = 0 if improved else self.stalled + 1
        self.history.append(row)
        self.pending = None
        self._persist(artifacts)
        artifacts.write_json(f"evaluations/{row['index']:04d}.json", row)
        self._reports(artifacts)
        artifacts.log(
            f"{row['candidate_id']} reward={reward:.2f} decision={decision} objectives={row.get('objectives')} "
            f"best={self.archive[0]['candidate_id'] if self.archive else None} stalled={self.stalled}/{self.settings.no_improvement_limit}",
            event="iteration_end",
        )

    def _reports(self, artifacts):
        artifacts.write_json(
            "ranking.json",
            {
                "objective": OBJECTIVE,
                "candidates": self.archive,
                "global_optimality_proven": False,
            },
        )
        artifacts.write_json("best.json", self.archive[0] if self.archive else None)
        columns = [
            "candidate_id",
            "parent_id",
            "direction_id",
            "decision",
            "reward",
            "objectives",
            "llm_seconds",
        ]
        import io

        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(columns)
        for row in self.history:
            writer.writerow(
                [
                    (
                        json.dumps(row[k], ensure_ascii=False)
                        if k == "objectives"
                        else row.get(k)
                    )
                    for k in columns
                ]
            )
        atomic_text(artifacts.root / "trajectory.csv", buffer.getvalue())
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "候选",
                "平均机巢数",
                "平均全场飞行时间/s",
                "每实例区域均时的平均/s",
                "最坏区域飞行时间/s",
            ]
        )
        for row in self.archive:
            m = row["evaluation"]["aggregate"]
            writer.writerow(
                [
                    row["candidate_id"],
                    m["mean_nest_count"],
                    m["mean_total_flight_seconds"],
                    m["mean_region_flight_seconds"],
                    m["maximum_flight_seconds"],
                ]
            )
        atomic_text(artifacts.root / "ranking.csv", "\ufeff" + buffer.getvalue())

    def run(self, run_id=None, *, resume=False, reset_stall=False):
        if resume and not run_id:
            raise ValueError("Resume requires a run ID.")
        run_id = (
            run_id
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid4().hex[:6]
        )
        with run_lock(self.outputs, run_id):
            return self._run_locked(run_id, resume=resume, reset_stall=reset_stall)

    def _run_locked(self, run_id, *, resume=False, reset_stall=False):
        self.bundle = prepare_data(self.settings)
        system = (
            Path(__file__).resolve().parents[1] / "prompts/system-v5.md"
        ).read_text(encoding="utf-8")
        system += (
            "\n工具接口：\n"
            + tool_reference()
            + "\n内置名称："
            + ", ".join(sorted(CANDIDATE_BUILTINS))
        )
        self.identity = self._identity(self.bundle, system)
        if not resume and run_id and run_path(self.outputs, run_id).exists():
            raise FileExistsError("Run exists; use a new ID or explicit resume.")
        if resume and (
            not run_id
            or not (run_path(self.outputs, run_id) / "search_state.json").is_file()
        ):
            raise FileNotFoundError("No resumable v5 state.")
        artifacts = RunArtifacts(self.outputs, run_id, secrets=(self.llm.api_key,))
        self.active_start = time.perf_counter()
        if resume:
            self._restore(artifacts)
        if reset_stall:
            self.stalled = 0
        artifacts.write_json("config_snapshot.json", self.identity)
        if hasattr(self.client, "set_observer"):
            self.client.set_observer(
                lambda message, **fields: self._observe_response(
                    artifacts, message, fields
                )
            )
        artifacts.log(
            f"Search v5 origin={'offline_mock' if self.mock else 'llm'} datasets={len(self.bundle['datasets'])} "
            f"completed={len(self.history)} budget={self.settings.evaluation_budget}; frozen greedy oracle={self.planner.metadata}",
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
                        "stage": "generation",
                    }
                    self._persist(artifacts)
                    artifacts.log(
                        f"round={index}/{self.settings.evaluation_budget} parent={self.pending['parent_id']} direction={direction} reason={reason}",
                        event="iteration_start",
                    )
                    start = time.perf_counter()
                    try:
                        artifacts.log(
                            f"{self.pending['candidate_id']} LLM request start",
                            event="llm_start",
                        )
                        with artifacts.heartbeat(
                            f"{self.pending['candidate_id']} LLM request active",
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
                                "Generated source contains sensitive-looking content."
                            )
                    except Exception as error:
                        self.pending.update(
                            llm_seconds=time.perf_counter() - start,
                            api_metadata=getattr(self.client, "last_metadata", {}),
                        )
                        artifacts.log(
                            f"{self.pending['candidate_id']} generation failed: {type(error).__name__}: {error}",
                            event="llm_error",
                        )
                        self._finish(
                            artifacts,
                            self.pending,
                            {
                                "status": "generation_error",
                                "feasible": False,
                                "violations": [f"{type(error).__name__}: {error}"],
                                "datasets": [],
                            },
                        )
                        continue
                    self.pending.update(
                        asdict(generated),
                        source_path=f"candidates/{index:04d}.py",
                        source_sha256=hashlib.sha256(
                            generated.algorithm_code.encode()
                        ).hexdigest(),
                        source_fingerprint=fingerprint(generated.algorithm_code),
                        llm_seconds=time.perf_counter() - start,
                        api_metadata=getattr(self.client, "last_metadata", {}),
                        stage="source_ready",
                    )
                    self._persist(artifacts)
                    artifacts.write_text(
                        self.pending["source_path"], generated.algorithm_code
                    )
                    self.pending.pop("algorithm_code")
                    self.pending["stage"] = "evaluation"
                    self._log_candidate(
                        artifacts, self.pending, generated.algorithm_code
                    )
                    artifacts.log(
                        f"{self.pending['candidate_id']} LLM finished in {self.pending['llm_seconds']:.3f}s",
                        event="llm_end",
                    )
                    self._persist(artifacts)
                row = self.pending
                if row["stage"] == "generation":
                    self._finish(
                        artifacts,
                        row,
                        {
                            "status": "generation_error",
                            "feasible": False,
                            "datasets": [],
                            "violations": [
                                "Interrupted provider request; consumed without automatic replay."
                            ],
                        },
                    )
                    continue
                try:
                    evaluation = self._evaluate(artifacts, row)
                except InfrastructureError:
                    raise
                except TimeoutError as error:
                    evaluation = {
                        "status": "candidate_timeout",
                        "feasible": False,
                        "datasets": row.get("completed_datasets", []),
                        "failure_stage": row.get("evaluation_phase"),
                        "violations": [str(error)],
                    }
                    artifacts.log(
                        f"{row['candidate_id']} candidate deadline exceeded; worker response settled",
                        event="candidate_timeout",
                    )
                except (MemoryError, OSError) as error:
                    raise InfrastructureError(
                        f"Host infrastructure failure: {type(error).__name__}: {error}"
                    ) from error
                except Exception as error:
                    evaluation = {
                        "status": (
                            error.status
                            if isinstance(error, CandidateExecutionError)
                            else "candidate_error"
                        ),
                        "feasible": False,
                        "datasets": row.get("completed_datasets", []),
                        "failure_stage": row.get("evaluation_phase"),
                        "violations": [f"{type(error).__name__}: {str(error)[:2000]}"],
                    }
                    if isinstance(error, CandidateExecutionError):
                        evaluation.update(
                            contract_errors=error.telemetry.get("contract_errors", []),
                            telemetry=error.telemetry,
                        )
                    artifacts.log(
                        f"{row['candidate_id']} phase={row.get('evaluation_phase')} error={error}",
                        event="candidate_error",
                    )
                self._finish(artifacts, row, evaluation)
        except BaseException as error:
            self._persist(artifacts)
            artifacts.log(
                f"Stopped safely: {type(error).__name__}: {error}; pending source preserved, no automatic API replay.",
                event=(
                    "infrastructure_error"
                    if isinstance(error, InfrastructureError)
                    else "interrupted"
                ),
            )
            raise
        reason = (
            "budget_exhausted"
            if len(self.history) >= self.settings.evaluation_budget
            else "best_stalled"
        )
        artifacts.write_json(
            "summary.json",
            {
                "search_version": 5,
                "tool_contract_version": 4,
                "origin": "offline_mock" if self.mock else "llm",
                "status": (
                    "best_within_budget"
                    if self.archive
                    else "no_deployable_llm_candidate"
                ),
                "stop_reason": reason,
                "evaluated_rounds": len(self.history),
                "best_candidate": (
                    self.archive[0]["candidate_id"] if self.archive else None
                ),
                "fully_evaluated_candidates": sum(
                    len(r["evaluation"].get("datasets", []))
                    == len(self.bundle["datasets"])
                    for r in self.history
                ),
                "global_optimality_proven": False,
                "oracle": self.planner.metadata,
                "oracle_counters": self.planner.stats(),
                "active_search_seconds": self.elapsed_seconds
                + time.perf_counter()
                - self.active_start,
            },
        )
        self._reports(artifacts)
        self._persist(artifacts)
        artifacts.log(
            f"Search complete stop={reason} best={self.archive[0]['candidate_id'] if self.archive else None}",
            event="search_end",
        )
        return artifacts.root
