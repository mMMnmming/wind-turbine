"""The executable policy cannot import evaluator, Stage2, files or processes."""
from __future__ import annotations

import ast
import symtable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCandidate:
    algorithm_code: str
    algorithm_idea: str
    change_hypothesis: str
    expected_improvement: str
    change_summary: str
    direction_id: str


BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "len", "list", "map", "max", "min",
    "next", "range", "reversed", "round", "set", "sorted", "str", "sum",
    "tuple", "zip", "Exception", "ValueError", "KeyError", "RuntimeError",
    "TypeError", "IndexError", "StopIteration", "ZeroDivisionError",
}
IMPORTS = {"math", "random", "itertools", "collections", "typing", "candidate_ops"}
SIGNATURES = {
    "initialize": ["problem", "initial_solution", "seed"],
    "propose": ["problem", "current_solution", "state", "feedback"],
}


def validate_candidate_source(source):
    if not isinstance(source, str) or not 1 <= len(source) <= 100000:
        raise ValueError("Source length must be 1..100000")
    tree = ast.parse(source, filename="<candidate>")
    for name, expected in SIGNATURES.items():
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(matches) != 1:
            raise ValueError("Define exactly one synchronous " + name + str(tuple(expected)))
        args = matches[0].args
        if ([a.arg for a in args.posonlyargs + args.args] != expected or
                args.vararg or args.kwarg or args.kwonlyargs or args.defaults):
            raise ValueError("Invalid signature for " + name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise ValueError("Use synchronous functions and explicit JSON state, without classes/global/nonlocal")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Private attribute forbidden at line {node.lineno}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names forbidden")
        if isinstance(node, ast.Import):
            if any(n.name not in IMPORTS for n in node.names):
                raise ValueError("Import outside policy allowlist")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in IMPORTS or any(n.name.startswith("_") or n.name == "*" for n in node.names):
                raise ValueError("Only explicit public imports from allowed modules")
    root = symtable.symtable(source, "<candidate>", "exec")
    tables = [root]
    for table in tables:
        tables.extend(table.get_children())
    defined = {s.get_name() for s in root.get_symbols()
               if s.is_assigned() or s.is_imported() or s.is_namespace()}
    unknown = sorted({s.get_name() for t in tables for s in t.get_symbols()
                      if s.is_referenced() and s.is_global() and
                      s.get_name() not in defined | BUILTINS})
    if unknown:
        raise ValueError("Unavailable names: " + ", ".join(unknown))
