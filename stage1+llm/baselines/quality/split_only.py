"""Hand-written engineering reference, NOT a generated or optimized LLM result."""

from candidate_tools import split_repair


def solve(turbines, config, seed):
    return split_repair({t.identifier: turbines[0].identifier for t in turbines})
