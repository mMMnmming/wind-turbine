"""Lifetime execution evidence, independent of repair-window counters."""

import copy
import hashlib
import json
from collections import Counter


def assignment_hash(assignments):
    return hashlib.sha256(
        json.dumps(assignments, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class SearchEvidence:
    def __init__(self):
        self.seen = set()
        self.feasible = set()
        self.values = {}
        self.first = self.best = self.saved = self.returned = None
        self.requests = self.invalid = self.repeats = self.improvements = 0
        self.fixed_attempts = self.fixed_improvements = self.saves = 0
        self.failures = Counter()
        self.move_seen = set()
        self.fixed_moves = set()

    def observe_move(self, before, after):
        if before == after:
            return
        digest = assignment_hash(after)
        self.move_seen.add(digest)
        if len(set(before.values())) == len(set(after.values())):
            self.fixed_moves.add(digest)

    def observe(self, proposal, checked):
        self.requests += 1
        for d in checked["diagnostics"]:
            self.failures[d["kind"]] += 1
        if checked["metrics"] is None:
            self.invalid += 1
            return
        digest = assignment_hash(proposal["assignments"])
        if digest in self.seen:
            self.repeats += 1
            return
        self.seen.add(digest)
        fixed = (
            self.best is not None
            and checked["metrics"]["nest_count"] == self.best["objectives"][0]
        )
        self.fixed_attempts += int(fixed)
        if not checked["feasible"]:
            return
        record = {
            "assignment_sha256": digest,
            "objectives": list(checked["objectives"]),
        }
        self.feasible.add(digest)
        self.values[digest] = record
        if self.first is None:
            self.first = record
        if self.best is not None and tuple(record["objectives"]) < tuple(
            self.best["objectives"]
        ):
            self.improvements += 1
            self.fixed_improvements += int(fixed)
        if self.best is None or tuple(record["objectives"]) < tuple(
            self.best["objectives"]
        ):
            self.best = record

    def mark(self, proposal, returned=False):
        digest = assignment_hash(proposal["assignments"])
        record = self.values.get(
            digest, {"assignment_sha256": digest, "objectives": None}
        )
        if returned:
            self.returned = record
        else:
            self.saves += int(self.saved != record)
            self.saved = record

    def snapshot(self):
        return copy.deepcopy(
            {
                "deployment_requests": self.requests,
                "invalid_structure_requests": self.invalid,
                "distinct_structural_deployments": len(self.seen),
                "distinct_feasible_deployments": len(self.feasible),
                "repeated_deployment_requests": self.repeats,
                "strict_improvements": self.improvements,
                "fixed_nest_distinct_attempts": len(self.fixed_moves),
                "structural_attempt_scope": "explicit structural primitives only; self-coded unobserved moves excluded",
                "distinct_structural_moves": len(self.move_seen),
                "fixed_nest_distinct_deployment_checks": self.fixed_attempts,
                "fixed_nest_strict_improvements": self.fixed_improvements,
                "distinct_consecutive_feasible_saves": self.saves,
                "first_feasible": self.first,
                "best_observed": self.best,
                "last_remembered": self.saved,
                "returned": self.returned,
                "failure_counts": dict(self.failures),
            }
        )
