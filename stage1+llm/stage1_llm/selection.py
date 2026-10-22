"""Discounted UCB selection over code-edit operators."""

from __future__ import annotations
import math
import random


class DiscountedUCB:
    def __init__(
        self,
        arms: list[str],
        decay: float = 0.9,
        exploration: float = 0.7,
        seed: int = 1234,
    ):
        if not arms or not 0 < decay <= 1 or exploration < 0:
            raise ValueError("Invalid DiscountedUCB parameters.")
        self.arms, self.decay, self.exploration = list(arms), decay, exploration
        self.counts, self.values, self.total = (
            {a: 0 for a in arms},
            {a: 0.0 for a in arms},
            0,
        )
        self.rng = random.Random(seed)

    def select_with_trace(
        self, forced_arm: str | None = None
    ) -> tuple[str, dict[str, dict[str, float | int | bool]]]:
        """Return an arm and the full score decomposition used to choose it."""
        if forced_arm is not None:
            if forced_arm not in self.arms:
                raise KeyError(forced_arm)
            return forced_arm, {
                name: {
                    "forced": name == forced_arm,
                    "value": self.values[name],
                    "count": self.counts[name],
                    "exploration": 0.0,
                    "score": self.values[name],
                }
                for name in self.arms
            }
        unexplored = [a for a in self.arms if self.counts[a] == 0]
        if unexplored:
            selected = unexplored[0]
            return selected, {
                name: {
                    "forced": False,
                    "value": self.values[name],
                    "count": self.counts[name],
                    "exploration": 1e308 if name == selected else 0.0,
                    "score": 1e308 if name == selected else self.values[name],
                }
                for name in self.arms
            }
        trace = {}
        for name in self.arms:
            bonus = self.exploration * math.sqrt(
                math.log(self.total + 1) / self.counts[name]
            )
            trace[name] = {
                "forced": False,
                "value": self.values[name],
                "count": self.counts[name],
                "exploration": bonus,
                "score": self.values[name] + bonus,
            }
        best = max(float(row["score"]) for row in trace.values())
        return (
            self.rng.choice(
                [name for name, row in trace.items() if row["score"] == best]
            ),
            trace,
        )

    def select(self) -> str:
        return self.select_with_trace()[0]

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.values:
            raise KeyError(arm)
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite.")
        for name in self.arms:
            self.values[name] *= self.decay
        self.values[arm] += (1.0 - self.decay) * reward
        self.counts[arm] += 1
        self.total += 1

    def state_dict(self) -> dict:
        return {
            "arms": self.arms,
            "decay": self.decay,
            "exploration": self.exploration,
            "counts": self.counts,
            "values": self.values,
            "total": self.total,
            "rng_state": _to_json(self.rng.getstate()),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state["arms"] != self.arms
            or state["decay"] != self.decay
            or state["exploration"] != self.exploration
        ):
            raise ValueError(
                "Saved UCB settings do not match the current search configuration."
            )
        self.counts = {arm: int(state["counts"][arm]) for arm in self.arms}
        self.values = {arm: float(state["values"][arm]) for arm in self.arms}
        self.total = int(state["total"])
        self.rng.setstate(_from_json(state["rng_state"]))


def _to_json(value):
    return [_to_json(item) for item in value] if isinstance(value, tuple) else value


def _from_json(value):
    return (
        tuple(_from_json(item) for item in value) if isinstance(value, list) else value
    )


class SlidingWindowUCB:
    """Both empirical means and counts use the same last W credited rounds."""

    def __init__(
        self,
        arms: list[str],
        window: int = 32,
        exploration: float = 0.7,
        seed: int = 1234,
    ):
        if (
            not arms
            or len(set(arms)) != len(arms)
            or type(window) is not int
            or window < 1
        ):
            raise ValueError("Invalid arms or window.")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("Invalid exploration coefficient.")
        self.arms, self.window, self.exploration = list(arms), window, exploration
        self.history: list[tuple[str, float]] = []
        self.total = 0
        self.rng = random.Random(seed)

    @property
    def counts(self):
        return {a: sum(arm == a for arm, _ in self.history) for a in self.arms}

    def select_with_trace(self, forced_arm=None, *, eligible=None):
        allowed = self.arms if eligible is None else list(eligible)
        if not allowed or not set(allowed) <= set(self.arms):
            raise ValueError("No eligible arms or unknown arm.")
        if forced_arm is not None and forced_arm not in allowed:
            raise ValueError("Forced arm is not eligible.")
        trace = {}
        counts = self.counts
        for arm in self.arms:
            count = counts[arm]
            mean = sum(r for a, r in self.history if a == arm) / count if count else 0.0
            bonus = (
                self.exploration * math.sqrt(math.log(len(self.history) + 1) / count)
                if count
                else 1e6
            )
            trace[arm] = {
                "count": count,
                "value": mean,
                "exploration": bonus,
                "score": mean + bonus,
                "eligible": arm in allowed,
                "forced": arm == forced_arm,
            }
        if forced_arm is not None:
            return forced_arm, trace
        highest = max(trace[a]["score"] for a in allowed)
        return (
            self.rng.choice([a for a in allowed if trace[a]["score"] == highest]),
            trace,
        )

    def update(self, arm, reward):
        if arm not in self.arms or not math.isfinite(reward):
            raise ValueError("Invalid arm/reward.")
        self.history = [*self.history, (arm, float(reward))][-self.window :]
        self.total += 1

    def state_dict(self):
        return {
            "kind": "sliding_window",
            "arms": self.arms,
            "window": self.window,
            "exploration": self.exploration,
            "history": self.history,
            "total": self.total,
            "rng_state": _to_json(self.rng.getstate()),
        }

    def load_state_dict(self, state):
        if (
            state.get("kind"),
            state["arms"],
            state["window"],
            state["exploration"],
        ) != ("sliding_window", self.arms, self.window, self.exploration):
            raise ValueError("Saved sliding-window settings mismatch.")
        self.history = [(a, float(r)) for a, r in state["history"]]
        if len(self.history) > self.window or any(
            a not in self.arms or not math.isfinite(r) for a, r in self.history
        ):
            raise ValueError("Corrupt window state.")
        self.total = int(state["total"])
        self.rng.setstate(_from_json(state["rng_state"]))
