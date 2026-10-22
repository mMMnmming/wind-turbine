"""Candidate source and proposal contracts."""

from __future__ import annotations
import ast
import symtable
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateProposal:
    nest_ids: list[str]
    assignments: dict[str, str]


@dataclass(frozen=True)
class ModelCandidate:
    algorithm_code: str
    algorithm_idea: str
    change_hypothesis: str
    expected_improvement: str
    change_summary: str
    direction_id: str


_ALLOWED_IMPORTS = {
    "math",
    "random",
    "itertools",
    "collections",
    "typing",
    "candidate_tools",
}

# One contract shared by source validation and the isolated worker. Reflective
# reads are wrappers, never the unrestricted builtins with the same names.
CANDIDATE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "ValueError",
        "KeyError",
        "RuntimeError",
        "TypeError",
        "AttributeError",
        "getattr",
        "hasattr",
    }
)


def validate_candidate_source(source: str) -> None:
    if not source.strip() or len(source) > 100_000:
        raise ValueError("Candidate source must contain 1..100000 characters.")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(
            f"Candidate source has invalid Python syntax: {error}"
        ) from error
    solvers = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve"
    ]
    if not solvers:
        raise ValueError(
            "Candidate must define top-level solve(turbines, config, seed)."
        )
    solver = solvers[0]
    if isinstance(solver, ast.AsyncFunctionDef):
        raise ValueError("solve must be a regular synchronous function.")
    positional = [arg.arg for arg in solver.args.posonlyargs + solver.args.args]
    if (
        positional[:3] != ["turbines", "config", "seed"]
        or len(positional) != 3
        or solver.args.vararg
        or solver.args.kwarg
    ):
        raise ValueError(
            "solve signature must be exactly solve(turbines, config, seed)."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            raise ValueError(
                "Candidate must use functions, not classes, inside the restricted runner."
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [item.name.split(".")[0] for item in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            if any(name not in _ALLOWED_IMPORTS for name in names):
                raise ValueError("Candidate imports a module outside the allowlist.")
            if isinstance(node, ast.ImportFrom) and any(
                item.name == "*" for item in node.names
            ):
                raise ValueError(
                    "Wildcard imports are not supported; import explicit public names."
                )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are not allowed in candidate source.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            excerpt = source.splitlines()[node.lineno - 1].strip()[:180]
            raise ValueError(
                f"Private and dunder attribute access is not allowed: {node.attr} at line {node.lineno}: {excerpt}"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {"eval", "exec", "compile", "open", "__import__", "input"}
        ):
            raise ValueError(f"Candidate uses forbidden builtin: {node.func.id}.")
    # Python's scope analysis distinguishes globals from parameters, closures,
    # comprehensions, aliases and local imports. This does not prove all runtime
    # paths valid (e.g. conditional local initialization still needs preflight).
    try:
        root = symtable.symtable(source, "<candidate>", "exec")
    except SyntaxError as error:
        raise ValueError(f"Candidate has invalid Python scope: {error}") from error
    tables = [root]
    for table in tables:
        tables.extend(table.get_children())
    defined = {
        symbol.get_name()
        for symbol in root.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    defined.update(
        symbol.get_name()
        for table in tables
        for symbol in table.get_symbols()
        if symbol.is_global() and (symbol.is_assigned() or symbol.is_imported())
    )
    available = defined | CANDIDATE_BUILTINS | {"math", "random"}
    unknown = sorted(
        {
            symbol.get_name()
            for table in tables
            for symbol in table.get_symbols()
            if symbol.is_referenced()
            and symbol.is_global()
            and symbol.get_name() not in available
        }
    )
    if unknown:
        raise ValueError(
            f"Names not available in candidate runtime: {', '.join(unknown)}. "
            "Use the documented builtins or define/import the helper explicitly."
        )
