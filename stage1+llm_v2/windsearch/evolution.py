"""Code evolution, honest failures, compact feedback and crash-safe paid rounds."""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .artifacts import Records, sha256_file, signature
from .candidate import validate_candidate_source
from .client import DeepSeekClient, parse_model_response
from .planner import InfrastructureError
from .runtime import run_instance
from .settings import LOCAL

OPERATORS = {
    "initialize": "从零设计完整搜索策略，从单风机单巢逐步优化，优先确保实际运行。",
    "structural_mutation": "修改搜索机制或控制流程，提出可由实际成绩验证的结构变化。",
    "elite_refinement": "保留父代有效机制，针对真实瓶颈改进完整策略。",
    "parameter_refinement": "保留父代基本结构，仅改进内部参数、候选规模或预算分配。",
    "parent_fusion": "对比两个父代的代码与成绩，组合互补有效机制，不盲目拼接。",
    "repair": "针对最近的具体代码错误修复完整策略，保留已有有效机制。",
}


@contextmanager
def exclusive_run(root):
    """OS releases the lock on exit/crash; stale files do not block resume."""
    path = Path(root) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This run is already active") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This run is already active") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def compact_instance(row):
    return {k: row[k] for k in ("name", "split", "code_success", "feasible", "stop_reason",
            "only_initial_solution", "initial_objectives", "objectives", "metrics", "search",
            "failure_counts", "timing", "oracle_counters", "final_audit_counters")}


def aggregate(rows):
    return [sum(r["metrics"]["nest_count"] for r in rows) / len(rows),
            sum(r["metrics"]["total_flight_seconds"] for r in rows) / len(rows),
            max(r["metrics"]["maximum_flight_seconds"] for r in rows)]


class Evolution:
    def __init__(self, settings, llm, bundle, planner, records):
        self.settings, self.llm, self.bundle = settings, llm, bundle
        self.planner, self.records = planner, records
        self.history = []
        self.pending = None
        self.started = time.perf_counter()
        self.elapsed_before = 0.0
        self.client = None
        self.system = (LOCAL / "prompts/system.md").read_text(encoding="utf-8")
        self.identity = {"version": 2, "interface": "initialize-propose-v2",
                         "settings": settings.snapshot(), "dataset": bundle["fingerprint"],
                         "oracle": planner.metadata, "provider": {
                             k: v for k, v in asdict(llm).items() if k != "api_key"},
                         "runtime_sources": {p.name: sha256_file(p) for p in
                                             sorted(Path(__file__).parent.glob("*.py"))},
                         "prompt_sha256": signature(self.system)}

    def save(self):
        self.records.json("diagnostics/runtime/search_state.json", {
            "identity": self.identity, "history": self.history, "pending": self.pending,
            "active_seconds": self.elapsed_before + time.perf_counter() - self.started})

    def archive(self):
        return sorted((r for r in self.history if r.get("eligible")),
                      key=lambda r: tuple(r["objectives"]))[:self.settings.archive_size]

    def choose(self, index):
        archive = self.archive()
        if not self.history:
            return "initialize", []
        # Carry the actual failing source and error into the next paid round.
        if self.history[-1].get("error"):
            return "repair", [self.history[-1]]
        if not archive:
            return "initialize", []
        op = ["structural_mutation", "elite_refinement", "parameter_refinement", "parent_fusion"][(index-2) % 4]
        if op == "parent_fusion" and len(archive) < 2:
            op = "structural_mutation"
        parents = archive[:2] if op == "parent_fusion" else archive[:1]
        return op, parents

    def feedback(self, op, parents):
        context = {"direction_id": op, "instruction": OPERATORS[op],
                   "initialization": "every candidate-instance starts with one nest per turbine",
                   "work_limits": {k: getattr(self.settings, k) for k in
                                   ("search_seconds", "max_proposals", "max_stage2_inferences", "stagnation_limit")},
                   "parents": [], "recent": [], "best_objectives": (
                       self.archive()[0]["objectives"] if self.archive() else None)}
        for parent in parents:
            context["parents"].append({"candidate_id": parent["candidate_id"],
                "algorithm_code": (self.records.root / parent["source_path"]).read_text(encoding="utf-8")
                    if parent.get("source_path") else None,
                "idea": parent.get("algorithm_idea"), "objectives": parent.get("objectives"),
                "instances": [self.feedback_instance(i) for i in parent.get("instances", [])],
                "error": self.error_brief(parent.get("error")),
                "timing": parent.get("timing")})
        for row in self.history[-3:]:
            context["recent"].append({"candidate_id": row["candidate_id"],
                "change_hypothesis": row.get("change_hypothesis"), "status": row["status"],
                "objectives": row.get("objectives"), "improved_parent": row.get("improved_parent"),
                "error": self.error_brief(row.get("error")),
                "evidence_label": "observed_gain" if row.get("improved_parent") else
                    ("explicit_error" if row.get("error") else "no_gain_observed_this_run")})
        return context

    @staticmethod
    def feedback_instance(row):
        return {"name": row["name"], "objectives": row["objectives"],
                "minimum_endurance_margin_seconds": row["metrics"]["minimum_endurance_margin_seconds"],
                "stop_reason": row["stop_reason"], "only_initial_solution": row["only_initial_solution"],
                "search_seconds": row["timing"]["search_seconds"],
                "strategy_seconds": row["timing"]["strategy_seconds"],
                "evaluation_seconds": row["timing"]["evaluation_seconds"],
                "inferences": row["oracle_counters"].get("inferences", 0),
                "search": {k: row["search"].get(k) for k in (
                    "distinct_candidates", "feasible_candidates", "improvements", "fixed_k_improvements",
                    "duplicates", "first_improvement_seconds", "best_at_seconds")},
                "failure_counts": row["failure_counts"]}

    @staticmethod
    def error_brief(error):
        if not error:
            return None
        return {k: error[k] for k in ("error", "error_type", "candidate_frames", "instance") if k in error}

    def generate(self, row, context):
        cid = row["candidate_id"]
        self.records.json("llm_feedback/" + cid + ".json", context)
        self.records.json("diagnostics/requests/" + cid + ".json", {
            "system": self.system, "feedback": context,
            "provider": self.identity["provider"]})
        self.pending = {"phase": "request_started", "row": row}
        self.save()  # Write before dispatch: unknown requests are never replayed.
        self.records.progress(f"[第{row['index']}/{self.settings.evaluation_budget}代] "
                              f"{row['operator']}｜父代{','.join(row['parent_ids']) or '无，从零生成'}｜生成代码中")
        stop = threading.Event()
        start = time.perf_counter()

        def pulse():
            while not stop.wait(self.settings.heartbeat_seconds):
                self.records.progress(f"[生成中 {cid}] 等待LLM完整响应｜耗时{time.perf_counter()-start:.0f}s")

        thread = threading.Thread(target=pulse, daemon=True)
        thread.start()
        try:
            if self.client is None:
                self.client = DeepSeekClient(self.llm)
                self.client.set_observer(lambda message, **fields: self.records.event(
                    "provider_event", candidate=self.pending["row"]["candidate_id"],
                    message=message, **fields))
            model = self.client.generate(self.system, json.dumps(context, ensure_ascii=False),
                                         expected_direction=row["operator"])
            row.update(asdict(model))
            row["source_path"] = "candidates/" + cid + ".py"
            self.records.text(row["source_path"], model.algorithm_code)
            row.pop("algorithm_code")
            row["source_sha256"] = sha256_file(self.records.root / row["source_path"])
            self.pending = {"phase": "evaluate", "row": row}
            self.save()
        finally:
            stop.set()
            thread.join(timeout=2)
            row["llm_seconds"] = time.perf_counter() - start
            if self.client is not None:
                row["api_metadata"] = self.client.last_metadata
                self.records.json("diagnostics/responses/" + cid + ".json", {
                    **self.client.last_response, "metadata": self.client.last_metadata})
            self.save()
        self.records.progress(f"[生成完成 {cid}] 耗时{row['llm_seconds']:.1f}s｜开始运行完整策略")

    def evaluate(self, row):
        source = (self.records.root / row["source_path"]).read_text(encoding="utf-8")
        validate_candidate_source(source)
        instances = row.setdefault("instances", [])
        if row.get("error"):
            row.update(eligible=False, objectives=None, status="candidate_error")
            return
        for index, dataset in enumerate(self.bundle["datasets"], 1):
            if any(r["name"] == dataset["name"] for r in instances):
                continue
            result = run_instance(source, str(self.records.root / row["source_path"]), dataset,
                                  self.settings, self.planner, self.records, row["candidate_id"],
                                  index, len(self.bundle["datasets"]))
            instances.append(compact_instance(result))
            if result["error"]:
                row["error"] = {**result["error"], "instance": dataset["name"]}
                self.save()
                break
            self.save()
        complete = len(instances) == len(self.bundle["datasets"])
        row["eligible"] = complete and not row.get("error") and all(
            r["code_success"] and r["feasible"] for r in instances)
        row["objectives"] = aggregate(instances) if row["eligible"] else None
        row["status"] = "completed" if row["eligible"] else "candidate_error"
        if instances:
            row["timing"] = {k: sum(r["timing"][k] for r in instances)
                             for k in instances[0]["timing"]}

    def finish(self, row):
        archive = self.archive()
        parent = next((r for r in self.history if r["candidate_id"] in row["parent_ids"]
                       and r.get("eligible")), None)
        key = tuple(row["objectives"]) if row.get("eligible") else None
        row["improved_parent"] = bool(key is not None and parent and key < tuple(parent["objectives"]))
        row["improved_global"] = bool(key is not None and (not archive or key < tuple(archive[0]["objectives"])))
        self.records.json("diagnostics/evaluations/" + row["candidate_id"] + ".json", row)
        self.history.append(row)
        self.pending = None
        self.save()
        self.reports("running")
        if row.get("eligible"):
            k, total, maximum = row["objectives"]
            self.records.progress(f"[本代结束 {row['candidate_id']}] {len(row['instances'])}/{len(self.bundle['datasets'])}实例正常可行"
                                  f"｜平均{k:.4f}巢 / {total:.1f}s｜最长区域{maximum:.1f}s"
                                  f"｜超过父代:{'是' if row['improved_parent'] else '否/无可行父代'}"
                                  f"｜刷新最好:{'是' if row['improved_global'] else '否'}")
        else:
            self.records.progress(f"[本代失败 {row['candidate_id']}] "
                                  + str((row.get('error') or {}).get('error', row['status']))[:200])

    def reports(self, status, error=None):
        archive = self.archive()
        best = archive[0] if archive else None
        summary = {"version": 2, "status": status, "evaluated_rounds": len(self.history),
                   "evaluation_budget": self.settings.evaluation_budget,
                   "fully_evaluated_candidates": sum(r.get("eligible", False) for r in self.history),
                   "best_candidate": best["candidate_id"] if best else None,
                   "best_objectives": best["objectives"] if best else None,
                   "oracle": self.planner.metadata,
                   "cache_policy": "cold_candidate_instance; separate cold final audit",
                   "initialization": "one_nest_per_turbine_every_candidate_instance",
                   "active_seconds": self.elapsed_before + time.perf_counter() - self.started,
                   "stop_reason": status, "error": self.error_brief(error),
                   "global_optimality_proven": False,
                   "total_llm_seconds": sum(r.get("llm_seconds", 0) for r in self.history),
                   "total_search_seconds": sum(r.get("timing", {}).get("search_seconds", 0) for r in self.history),
                   "total_search_inferences": sum(i["oracle_counters"].get("inferences", 0)
                        for r in self.history for i in r.get("instances", []))}
        self.records.json("summary.json", summary)
        self.records.json("instance_metrics.json", [{"candidate_id": r["candidate_id"],
            "status": r["status"], "instances": r.get("instances", [])} for r in self.history])
        columns = ["candidate_id", "parent_ids", "operator", "status", "objectives", "llm_seconds",
                   "search_seconds", "strategy_seconds", "evaluation_seconds", "total_seconds",
                   "successful_instances", "initial_only_instances", "stage2_inferences",
                   "improved_parent", "improved_global", "error"]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for row in self.history:
            out = {k: row.get(k) for k in columns}
            out.update(row.get("timing", {}))
            out = {k: out.get(k) for k in columns}
            instances = row.get("instances", [])
            out["successful_instances"] = sum(r["code_success"] for r in instances)
            out["initial_only_instances"] = sum(r["only_initial_solution"] for r in instances)
            out["stage2_inferences"] = sum(r["oracle_counters"].get("inferences", 0) for r in instances)
            out["error"] = (row.get("error") or {}).get("error")
            for field in ("parent_ids", "objectives"):
                out[field] = json.dumps(out[field], ensure_ascii=False)
            writer.writerow(out)
        self.records.text("trajectory.csv", buffer.getvalue())
        self.records.json("diagnostics/runtime/ranking.json", [{"candidate_id": r["candidate_id"],
                            "objectives": r["objectives"]} for r in archive])
        self.records.json("diagnostics/runtime/best.json", best)

    def recover_request(self):
        row = self.pending["row"]
        response = self.records.root / "diagnostics/responses" / (row["candidate_id"] + ".json")
        if response.exists():
            saved = json.loads(response.read_text(encoding="utf-8"))
            metadata = saved.get("metadata", {})
            if saved.get("finish_reason") == "stop" and metadata.get("stream_complete", True):
                try:
                    model = parse_model_response(saved["content"], row["operator"])
                except (KeyError, ValueError) as exc:
                    row.update(status="generation_error", eligible=False,
                               error={"error": str(exc), "error_type": type(exc).__name__})
                    self.finish(row)
                    return
                row.update(asdict(model))
                row["source_path"] = "candidates/" + row["candidate_id"] + ".py"
                self.records.text(row["source_path"], row.pop("algorithm_code"))
                row["source_sha256"] = sha256_file(self.records.root / row["source_path"])
                row["llm_seconds"] = metadata.get("elapsed_seconds", 0)
                row["api_metadata"] = metadata
                self.pending["phase"] = "evaluate"
                self.save()
                return
        row.update(status="interrupted_request_unknown", eligible=False,
                   error={"error": "Interrupted provider request: outcome unknown; not replayed",
                          "error_type": "InterruptedRequest"})
        self.finish(row)

    def run(self, resume=False):
        state_path = self.records.root / "diagnostics/runtime/search_state.json"
        identity_verified = False
        try:
            if resume:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state["identity"] != self.identity:
                    raise ValueError("Resume identity changed (code/config/data/model/provider); start a new run")
                identity_verified = True
                self.history, self.pending = state["history"], state["pending"]
                self.elapsed_before = state.get("active_seconds", 0)
                if self.pending and self.pending["phase"] == "request_started":
                    self.recover_request()
            elif state_path.exists():
                raise FileExistsError("Run already exists; use resume")
            else:
                identity_verified = True
            self.records.json("diagnostics/runtime/identity.json", self.identity)
            self.save()
            self.reports("running")
            while len(self.history) < self.settings.evaluation_budget:
                if self.pending:
                    row = self.pending["row"]
                else:
                    index = len(self.history) + 1
                    op, parents = self.choose(index)
                    row = {"index": index, "candidate_id": f"C{index:04d}", "operator": op,
                           "parent_ids": [r["candidate_id"] for r in parents], "instances": []}
                    try:
                        self.generate(row, self.feedback(op, parents))
                    except Exception as exc:
                        row.update(status="generation_error", eligible=False,
                                   error={"error": str(exc), "error_type": type(exc).__name__,
                                          "traceback": traceback.format_exc()})
                        self.finish(row)
                        continue
                try:
                    self.evaluate(row)
                except InfrastructureError:
                    raise
                except Exception as exc:
                    row.update(status="candidate_error", eligible=False,
                               error={"error": str(exc), "error_type": type(exc).__name__,
                                      "traceback": traceback.format_exc()})
                self.finish(row)
            status = "budget_exhausted" if self.archive() else "no_successful_algorithm"
            self.reports(status)
            best = self.archive()[0]["candidate_id"] if self.archive() else "无正常完成算法"
            self.records.progress(f"[运行结束] 完成{len(self.history)}轮｜最好:{best}｜结果:{self.records.root}")
        except BaseException as exc:
            # Do not overwrite a different run's reports on identity mismatch.
            if identity_verified:
                self.save()
                self.reports("interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure_or_run_error",
                             {"error": str(exc), "error_type": type(exc).__name__})
                self.records.event("run_interrupted", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if self.client is not None:
                self.client.close()
