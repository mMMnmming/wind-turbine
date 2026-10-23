"""Persistent restricted strategy process. No evaluator or Stage2 is exposed."""
from __future__ import annotations

import builtins
import collections
import copy
import itertools
import json
import math
import random
import sys
import time
import traceback
import typing
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windsearch.candidate import BUILTINS, validate_candidate_source
from windsearch import ops


def send(value):
    text = json.dumps(value, allow_nan=False, ensure_ascii=False)
    if len(text.encode("utf-8")) > 4_000_000:
        raise ValueError("Strategy output/state exceeds 4 MB")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    filename = "<candidate>"
    start = None
    try:
        data = json.loads(sys.stdin.readline())
        filename = data["filename"]
        validate_candidate_source(data["source"])
        facade = ModuleType("candidate_ops")
        for name in ("merge", "relocate", "remove", "reassign"):
            setattr(facade, name, getattr(ops, name))
        modules = {"math": math, "random": random, "itertools": itertools,
                   "collections": collections, "typing": typing, "candidate_ops": facade}

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level or name not in modules:
                raise ImportError("Module not exposed to strategy")
            return modules[name]

        allowed = {name: getattr(builtins, name) for name in BUILTINS}
        allowed["__import__"] = safe_import
        scope = {"__builtins__": allowed}
        start = time.perf_counter()
        random.seed(data["seed"])
        exec(compile(data["source"], filename, "exec"), scope, scope)
        state = scope["initialize"](copy.deepcopy(data["problem"]),
                                     copy.deepcopy(data["initial"]), data["seed"])
        send({"ok": True, "kind": "initialized", "strategy_seconds": time.perf_counter() - start,
              "state": state})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            start = time.perf_counter()
            answer = scope["propose"](copy.deepcopy(data["problem"]),
                                       request["current_solution"], state, request["feedback"])
            if not isinstance(answer, dict) or set(answer) != {"assignments", "state", "phase", "done"}:
                raise ValueError("propose must return exactly assignments,state,phase,done")
            if type(answer["done"]) is not bool or not isinstance(answer["phase"], str):
                raise ValueError("done must be bool; phase must be str")
            if answer["assignments"] is not None and not isinstance(answer["assignments"], dict):
                raise ValueError("assignments must be a complete dict[str,str] or null")
            state = answer["state"]
            send({"ok": True, "kind": "proposal", "answer": answer,
                  "strategy_seconds": time.perf_counter() - start})
    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        candidate_frames = [{"line": f.lineno, "function": f.name}
                            for f in frames if f.filename == filename]
        send({"ok": False, "error": str(error), "error_type": type(error).__name__,
              "candidate_frames": candidate_frames, "traceback": traceback.format_exc(),
              "strategy_seconds": time.perf_counter() - start if start is not None else 0.0})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
