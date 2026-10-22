"""AST isolation, live tool telemetry, and hard timeout; not an OS sandbox."""

from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from .candidate import CANDIDATE_BUILTINS, CandidateProposal, validate_candidate_source
from .domain import DeploymentConfig, Turbine

_WORKER = r"""import json,math,random,itertools,collections,typing,sys,builtins
from types import SimpleNamespace,ModuleType
data=json.load(sys.stdin)
sys.path.insert(0,data['runtime'])
from stage1_llm.candidate_tools import CandidateTools,BudgetExceeded,PUBLIC_METHODS
from stage1_llm.domain import DeploymentConfig,Turbine
def progress(row):
    sys.stderr.write('STAGE1_TOOLS '+json.dumps(row,allow_nan=False)+'\n');sys.stderr.flush()
tools=CandidateTools([Turbine(**t) for t in data['turbines']],DeploymentConfig(**data['config']),
                     observer=progress,**data['limits'])
facade=ModuleType('candidate_tools')
for name in PUBLIC_METHODS: setattr(facade,name,getattr(tools,name))
facade.BudgetExceeded=BudgetExceeded
allowed={name:getattr(builtins,name) for name in data['builtins'] if name not in ('getattr','hasattr')}
mods={"math":math,"random":random,"itertools":itertools,"collections":collections,"typing":typing,"candidate_tools":facade}
def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
    if name not in mods or level: raise ImportError('module not allowed')
    return mods[name]
allowed['__import__']=safe_import
scope={'__builtins__':allowed,'math':math,'random':random}
class TurbineInput(dict):
    @property
    def identifier(self): return self['identifier']
    @property
    def x_km(self): return self['x_km']
    @property
    def y_km(self): return self['y_km']
C=SimpleNamespace(**data['config'])
config_fields=frozenset(data['config'])
turbine_fields=frozenset(('identifier','x_km','y_km'))
missing=object()
def public_getattr(obj,name,default=missing):
    if not isinstance(name,str): raise TypeError('attribute name must be a string')
    if name.startswith('_'): raise ValueError('private attribute reads are forbidden')
    if obj is C: fields=config_fields
    elif type(obj) is TurbineInput: fields=turbine_fields
    else: raise ValueError('getattr/hasattr only support public input config or turbine fields')
    if name in fields: return getattr(obj,name)
    if default is not missing: return default
    raise AttributeError('No public input field: '+name)
def public_hasattr(obj,name):
    try: public_getattr(obj,name)
    except AttributeError: return False
    return True
allowed['getattr']=public_getattr
allowed['hasattr']=public_hasattr
termination='returned'
try:
    exec(compile(data['source'],sys.argv[1],'exec'),scope,scope)
    answer=scope['solve']([TurbineInput(t) for t in data['turbines']],C,int(data['seed']))
except BudgetExceeded as error:
    answer=tools.best()
    if answer is None: raise RuntimeError('Budget exhausted without explicitly remembered feasible deployment') from error
    termination='budget_exhausted'
if not isinstance(answer,dict) or set(answer)!={'nest_ids','assignments'}: raise ValueError('solve must return exactly nest_ids and assignments')
json.dump({'proposal':answer,'telemetry':dict(tools.stats(),termination=termination)},sys.stdout,allow_nan=False)
"""


def _runtime(source_path: Path) -> Path:
    """Verify exported helper snapshots before using them."""
    selection = source_path.parent / "selection.json"
    if selection.is_file():
        record = json.loads(selection.read_text(encoding="utf-8"))
        hashes = record.get("runtime_hashes")
        if hashes:
            import hashlib

            actual = hashlib.sha256(
                source_path.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            if actual != record.get("source_sha256"):
                raise ValueError("Selected algorithm source hash changed.")
            root = source_path.parent / "runtime"
            for name, expected in hashes.items():
                target = root / "stage1_llm" / name
                if Path(name).name != name or target.suffix != ".py":
                    raise ValueError("Unsafe runtime manifest.")
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    raise ValueError("Exported trusted runtime hash changed.")
            if set(hashes) != {
                "__init__.py",
                "candidate_tools.py",
                "domain.py",
                "routing.py",
                "candidate.py",
                "sandbox.py",
            }:
                raise ValueError("Incomplete trusted runtime manifest.")
            return root
    return Path(__file__).resolve().parents[1]


def run_candidate(
    source_path: Path,
    turbines: list[Turbine],
    config: DeploymentConfig,
    seed: int,
    timeout_seconds: int,
    *,
    limits: dict | None = None,
    telemetry: dict | None = None,
    observer=None,
) -> CandidateProposal:
    source = source_path.read_text(encoding="utf-8")
    validate_candidate_source(source)
    if (
        not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Hard timeout must be positive and finite.")
    effective_limits = dict(limits or {})
    effective_limits["soft_seconds"] = min(
        effective_limits.get("soft_seconds", 120.0), timeout_seconds * 0.8
    )
    payload = {
        "turbines": [
            {"identifier": t.identifier, "x_km": t.x_km, "y_km": t.y_km}
            for t in turbines
        ],
        "config": {n: getattr(config, n) for n in config.__dataclass_fields__},
        "seed": seed,
        "builtins": sorted(CANDIDATE_BUILTINS),
        "source": source,
        "runtime": str(_runtime(source_path)),
        "limits": effective_limits,
    }
    output: list[str] = []
    errors: list[str] = []
    events: queue.Queue[dict] = queue.Queue()
    started = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-I", "-X", "utf8", "-c", _WORKER, str(source_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert (
        proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    )

    def drain(pipe, sink, observe=False):
        for line in pipe:
            if observe and line.startswith("STAGE1_TOOLS "):
                try:
                    events.put_nowait(json.loads(line[len("STAGE1_TOOLS ") :]))
                except json.JSONDecodeError:
                    sink.append(line)
            else:
                sink.append(line)

    threads = [
        threading.Thread(target=drain, args=(proc.stdout, output), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, errors, True), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        proc.stdin.write(json.dumps(payload, allow_nan=False))
        proc.stdin.close()
        while True:
            while not events.empty():
                row = events.get_nowait()
                if observer:
                    observer(row)
            remaining = timeout_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                if telemetry is not None:
                    telemetry.update(
                        termination="hard_timeout",
                        seconds=time.perf_counter() - started,
                    )
                raise TimeoutError(f"Candidate exceeded {timeout_seconds}s time limit.")
            try:
                proc.wait(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for thread in threads:
            thread.join()
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()
        while not events.empty():
            row = events.get_nowait()
            if observer:
                observer(row)
    if proc.returncode:
        raise RuntimeError(
            ("".join(errors) or f"candidate exited {proc.returncode}")[-3000:]
        )
    try:
        envelope = json.loads("".join(output))
        result = envelope["proposal"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Candidate returned invalid JSON.") from error
    if not isinstance(result.get("nest_ids"), list) or not isinstance(
        result.get("assignments"), dict
    ):
        raise ValueError(
            "Candidate output must contain list nest_ids and mapping assignments."
        )
    if telemetry is not None:
        telemetry.update(envelope["telemetry"])
    return CandidateProposal(
        [str(x) for x in result["nest_ids"]],
        {str(k): str(v) for k, v in result["assignments"].items()},
    )
