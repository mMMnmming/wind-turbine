"""Hand-written engineering reference, NOT a generated or optimized LLM result."""

from candidate_tools import repartition_repair


def solve(turbines, config, seed):
    return repartition_repair({t.identifier: turbines[0].identifier for t in turbines})
