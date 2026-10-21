"""Discounted UCB selection over code-edit operators."""
from __future__ import annotations
import math
import random

class DiscountedUCB:
    def __init__(self, arms: list[str], decay: float = 0.9, exploration: float = 0.7, seed: int = 1234):
        if not arms or not 0 < decay <= 1 or exploration < 0:
            raise ValueError("Invalid DiscountedUCB parameters.")
        self.arms, self.decay, self.exploration = list(arms), decay, exploration
        self.counts, self.values, self.total = {a: 0 for a in arms}, {a: 0.0 for a in arms}, 0
        self.rng = random.Random(seed)
    def select_with_trace(self, forced_arm: str | None = None) -> tuple[str, dict[str, dict[str, float | int | bool]]]:
        """Return an arm and the full score decomposition used to choose it."""
        if forced_arm is not None:
            if forced_arm not in self.arms:
                raise KeyError(forced_arm)
            return forced_arm, {name: {"forced": name == forced_arm, "value": self.values[name], "count": self.counts[name], "exploration": 0.0, "score": self.values[name]} for name in self.arms}
        unexplored = [a for a in self.arms if self.counts[a] == 0]
        if unexplored:
            selected = unexplored[0]
            return selected, {name: {"forced": False, "value": self.values[name], "count": self.counts[name], "exploration": 1e308 if name == selected else 0.0, "score": 1e308 if name == selected else self.values[name]} for name in self.arms}
        trace = {}
        for name in self.arms:
            bonus = self.exploration * math.sqrt(math.log(self.total + 1) / self.counts[name])
            trace[name] = {"forced": False, "value": self.values[name], "count": self.counts[name], "exploration": bonus, "score": self.values[name] + bonus}
        best = max(float(row["score"]) for row in trace.values())
        return self.rng.choice([name for name, row in trace.items() if row["score"] == best]), trace

    def select(self) -> str:
        return self.select_with_trace()[0]
    def update(self, arm: str, reward: float) -> None:
        if arm not in self.values: raise KeyError(arm)
        if not math.isfinite(reward): raise ValueError("Reward must be finite.")
        for name in self.arms: self.values[name] *= self.decay
        self.values[arm] += (1.0 - self.decay) * reward
        self.counts[arm] += 1
        self.total += 1
    def state_dict(self) -> dict:
        return {"arms": self.arms, "decay": self.decay, "exploration": self.exploration, "counts": self.counts, "values": self.values, "total": self.total, "rng_state": _to_json(self.rng.getstate())}

    def load_state_dict(self, state: dict) -> None:
        if state["arms"] != self.arms or state["decay"] != self.decay or state["exploration"] != self.exploration:
            raise ValueError("Saved UCB settings do not match the current search configuration.")
        self.counts = {arm:int(state["counts"][arm]) for arm in self.arms}
        self.values = {arm:float(state["values"][arm]) for arm in self.arms}
        self.total = int(state["total"])
        self.rng.setstate(_from_json(state["rng_state"]))

def _to_json(value):
    return [_to_json(item) for item in value] if isinstance(value, tuple) else value

def _from_json(value):
    return tuple(_from_json(item) for item in value) if isinstance(value, list) else value
