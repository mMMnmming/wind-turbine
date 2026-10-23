"""Trusted search loop: singleton start, strict acceptance, independent final audit."""
from __future__ import annotations

import threading
import time
from collections import Counter

from .artifacts import signature
from .evaluator import Evaluator, WorkBudgetExceeded, compact_feedback, problem_input, solution
from .execution import StrategyError, StrategyWorker
from .planner import InfrastructureError
from .sandbox import CandidateDeadlineExceeded


def run_instance(source, filename, dataset, settings, planner, records, candidate_id, index, count):
    from .data import turbines_from_record

    points = turbines_from_record(dataset)
    name = dataset["name"]
    scope = candidate_id + "/" + name
    # Cold candidate-instance start; cache persists only inside this instance search.
    planner.reset_cache()
    evaluator = Evaluator(points, settings.deployment, planner, records, scope,
                          settings.heartbeat_seconds)
    total_started = time.perf_counter()
    assignments = {p.identifier: p.identifier for p in points}
    stage = {"name": "验证单风机单巢初解"}
    stop_heartbeat = threading.Event()
    stats = {"proposals": 0, "distinct_candidates": 0, "feasible_candidates": 0,
             "improvements": 0, "fixed_k_improvements": 0, "nest_reductions": 0,
             "duplicates": 0, "strategy_seconds": 0.0,
             "first_improvement_seconds": None, "best_at_seconds": 0.0}
    failures = Counter()
    best, best_checked = None, None
    search_started = None
    self_reported_phase = ""
    call_started = None

    def progress():
        elapsed = time.perf_counter() - (search_started or total_started)
        metrics = best["metrics"] if best else None
        text = (f"[进度 {candidate_id}/{name}] {stage['name']}｜耗时{elapsed:.0f}s"
                + (f"｜最好{metrics['nest_count']}巢 / {metrics['total_flight_seconds']:.1f}s" if metrics else "")
                + f"｜不同候选{stats['distinct_candidates']}｜改善{stats['improvements']}次")
        if self_reported_phase:
            text += "｜策略自报:" + self_reported_phase[:40]
        records.progress(text)

    def pulse():
        while not stop_heartbeat.wait(settings.heartbeat_seconds):
            progress()

    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    worker = None
    error = None
    stop_reason = "budget_exhausted"
    search_seconds, evaluation_seconds = 0.0, 0.0
    search_oracle, audit_oracle = {}, {}
    curve = []
    try:
        initial_checked = evaluator.check(assignments)
        if not initial_checked["feasible"]:
            raise InfrastructureError("Singleton initial solution failed canonical validation")
        best_checked = initial_checked
        initial = best = solution(assignments, initial_checked)
        problem = problem_input(points, settings.deployment, planner.metadata)
        records.json("diagnostics/runtime/" + scope + "-problem.json", problem)
        records.progress(f"[实例 {index}/{count} {candidate_id}/{name}] 初始{len(points)}巢｜开始部署搜索")
        records.event("instance_start", candidate=candidate_id, instance=name,
                      initial=initial, cache_policy="cold_candidate_instance")
        search_started = time.perf_counter()
        deadline = search_started + settings.search_seconds
        evaluation_start = evaluator.seconds
        worker = StrategyWorker(records, scope)
        stage["name"] = "候选代码初始化中"
        call_started = time.perf_counter()
        ready = worker.exchange({"source": source, "filename": filename, "problem": problem,
                                 "initial": initial, "seed": settings.seed},
                                deadline, settings.heartbeat_seconds, lambda: None)
        stats["strategy_seconds"] += ready["strategy_seconds"]
        feedback = {"kind": "initial", "accepted": True, **compact_feedback(initial_checked)}
        seen = {signature(assignments): compact_feedback(initial_checked)}
        stagnant = 0
        for attempt in range(settings.max_proposals):
            if time.perf_counter() >= deadline:
                break
            if planner.stats().get("inferences", 0) >= settings.max_stage2_inferences:
                stop_reason = "inference_budget_exhausted"
                break
            stage["name"] = "候选代码执行中"
            call_started = time.perf_counter()
            feedback["remaining"] = {
                "seconds": max(0.0, deadline - time.perf_counter()),
                "proposals": settings.max_proposals - attempt,
                "stage2_inferences": settings.max_stage2_inferences - planner.stats().get("inferences", 0),
                "stagnation_remaining": settings.stagnation_limit - stagnant,
            }
            row = worker.exchange({"current_solution": best, "feedback": feedback},
                                  deadline, settings.heartbeat_seconds, lambda: None)
            stats["strategy_seconds"] += row["strategy_seconds"]
            answer = row["answer"]
            self_reported_phase = answer["phase"]
            stats["proposals"] += 1
            candidate = answer["assignments"]
            accepted = False
            duplicate = False
            if candidate is None:
                feedback = {"kind": "no_proposal", "accepted": False}
                stagnant += 1
            else:
                key = signature(candidate)
                duplicate = key in seen
                if duplicate:
                    stats["duplicates"] += 1
                    failures["duplicate"] += 1
                    feedback = {**seen[key], "kind": "duplicate", "accepted": False}
                    stagnant += 1
                else:
                    stats["distinct_candidates"] += 1
                    stage["name"] = "框架约束检查 / Stage2评价"
                    checked = evaluator.check(candidate, deadline=deadline,
                                              inference_limit=settings.max_stage2_inferences)
                    compact = compact_feedback(checked)
                    seen[key] = compact
                    if checked["feasible"]:
                        stats["feasible_candidates"] += 1
                        accepted = tuple(checked["objectives"]) < tuple(best["objectives"])
                    else:
                        failures.update(d["kind"] for d in checked["diagnostics"])
                    record_checked = {**checked, "regions": [records.blob(r) for r in checked["regions"]]}
                    checked_ref = records.blob(record_checked)
                    records.append("diagnostics/evaluations/" + scope + "-checks.jsonl",
                                   {"attempt": attempt + 1, "proposal": records.blob(candidate),
                                    "checked": checked_ref, "accepted": accepted})
                    feedback = {**compact, "kind": "evaluation", "accepted": accepted}
                    if accepted:
                        old_k = best["metrics"]["nest_count"]
                        best_checked = checked
                        best = solution(candidate, checked)
                        stats["improvements"] += 1
                        fixed_k = best["metrics"]["nest_count"] == old_k
                        stats["fixed_k_improvements" if fixed_k else "nest_reductions"] += 1
                        elapsed = time.perf_counter() - search_started
                        stats["best_at_seconds"] = elapsed
                        if stats["first_improvement_seconds"] is None:
                            stats["first_improvement_seconds"] = elapsed
                        improvement = {"elapsed_seconds": elapsed, "attempt": attempt + 1,
                                       "objectives": best["objectives"]}
                        curve.append(improvement)
                        records.append("diagnostics/evaluations/" + scope + "-improvements.jsonl", improvement)
                        # Every accepted solution is persisted before the next callback.
                        records.json("diagnostics/runtime/" + scope + "-best.json", best)
                        if not fixed_k and (best['metrics']['nest_count'] <= 10 or old_k == len(points)):
                            records.progress(f"[减巢 {candidate_id}/{name}] {old_k}→{best['metrics']['nest_count']}巢"
                                             f"｜总航时{best['metrics']['total_flight_seconds']:.1f}s")
                        stagnant = 0
                    else:
                        stagnant += 1
            records.event("proposal_step", candidate=candidate_id, instance=name,
                          attempt=attempt + 1, accepted=accepted, duplicate=duplicate,
                          self_reported_phase=self_reported_phase, best_objectives=best["objectives"])
            if answer["done"]:
                stop_reason = "strategy_declared_neighborhood_exhausted"
                break
            if stagnant >= settings.stagnation_limit:
                stop_reason = "stagnation_limit"
                break
        else:
            stop_reason = "proposal_budget_exhausted"
    except (CandidateDeadlineExceeded, TimeoutError):
        stop_reason = "time_budget_exhausted"
        if stage["name"] in ("候选代码初始化中", "候选代码执行中"):
            if call_started is not None:
                stats["strategy_seconds"] += time.perf_counter() - call_started
                stats["timed_out_callback_uses_wall_time"] = True
            error = {"error_type": "StrategyTimeout",
                     "error": "Strategy callback did not return before the search deadline"}
            stop_reason = "candidate_timeout"
    except WorkBudgetExceeded:
        stop_reason = "inference_budget_exhausted"
    except StrategyError as exc:
        error = exc.detail
        stats["strategy_seconds"] += error.get("strategy_seconds", 0.0)
        stop_reason = "candidate_error"
    except InfrastructureError:
        raise
    except Exception as exc:
        import traceback
        error = {"error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
        stop_reason = "candidate_error"
    finally:
        if search_started is not None:
            search_seconds = time.perf_counter() - search_started
            evaluation_seconds = evaluator.seconds - evaluation_start
        if worker:
            worker.close()
        stop_heartbeat.set()
        heartbeat.join(timeout=2)

    if best is None:
        raise InfrastructureError("Initial deployment could not be established")
    planner.settle()
    search_oracle = planner.stats()
    # A fresh, independent final audit is outside the strategy's search budget.
    audit_start = time.perf_counter()
    stage["name"] = "最终独立路线审核"
    stop_heartbeat.clear()
    audit_heartbeat = threading.Thread(target=pulse, daemon=True)
    audit_heartbeat.start()
    records.event("final_audit_start", candidate=candidate_id, instance=name)
    try:
        planner.reset_cache()
        final_checked = evaluator.check(best["assignments"])
    finally:
        stop_heartbeat.set()
        audit_heartbeat.join(timeout=2)
    audit_oracle = planner.stats()
    if not final_checked["feasible"] or final_checked["objectives"] != best["objectives"]:
        raise InfrastructureError("Independent final audit disagrees with the saved best deployment")
    if error:
        records.json("diagnostics/errors/" + scope + ".json", error)
    result = {
        "name": name, "split": dataset["split"], "candidate_id": candidate_id,
        "code_success": error is None, "feasible": True, "error": error,
        "stop_reason": stop_reason, "only_initial_solution": stats["improvements"] == 0,
        "initial_objectives": initial["objectives"], "objectives": best["objectives"],
        "metrics": final_checked["metrics"], "search": dict(stats),
        "failure_counts": dict(failures), "improvement_curve": curve,
        "timing": {"search_seconds": search_seconds, "strategy_seconds": stats["strategy_seconds"],
                   "evaluation_seconds": evaluation_seconds,
                   "supervisor_seconds": max(0.0, search_seconds - evaluation_seconds - stats["strategy_seconds"]),
                   "final_audit_seconds": time.perf_counter() - audit_start,
                   "total_seconds": time.perf_counter() - total_started},
        "oracle_counters": search_oracle, "final_audit_counters": audit_oracle,
        "solution": best, "final_audit": final_checked,
    }
    records.json("diagnostics/evaluations/" + scope + ".json", result)
    labels = {"time_budget_exhausted": "时间预算耗尽", "proposal_budget_exhausted": "候选预算耗尽",
              "inference_budget_exhausted": "推理预算耗尽", "stagnation_limit": "连续无改善",
              "candidate_error": "代码错误", "strategy_declared_neighborhood_exhausted": "策略声明邻域结束"}
    labels["candidate_timeout"] = "候选代码执行超时"
    records.progress(f"[实例结束 {candidate_id}/{name}] {best['metrics']['nest_count']}巢 / "
                     f"{best['metrics']['total_flight_seconds']:.1f}s｜停止:{labels[stop_reason]}"
                     f"｜搜索{search_seconds:.1f}s（策略{stats['strategy_seconds']:.1f}s/评价{evaluation_seconds:.1f}s）"
                     + ("｜仅保底" if stats["improvements"] == 0 else ""))
    if error:
        records.progress(f"[错误 {candidate_id}/{name}] {error.get('error_type', 'StrategyError')}: "
                         + str(error.get("error", ""))[:180])
    return result
