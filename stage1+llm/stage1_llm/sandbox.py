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


class CandidateExecutionError(RuntimeError):
    """Execution failure with structured diagnostics retained across processes."""

    def __init__(
        self,
        message: str,
        *,
        status: str = "tool_contract_error",
        telemetry: dict | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.telemetry = telemetry or {}


_WORKER = r"""import json,math,random,itertools,collections,typing,sys,builtins,inspect
from types import SimpleNamespace,ModuleType
data=json.load(sys.stdin)
sys.path.insert(0,data['runtime'])
from stage1_llm.candidate_tools import CandidateTools,BudgetExceeded,PUBLIC_METHODS
import stage1_llm.candidate_tools as tool_runtime
from stage1_llm.domain import DeploymentConfig,Turbine
def progress(row):
    sys.stderr.write('STAGE1_TOOLS '+json.dumps(row,allow_nan=False)+'\n');sys.stderr.flush()
contract_version=getattr(tool_runtime,'TOOLS_VERSION',1)
extra={'candidate_filename':sys.argv[1]} if contract_version>=2 else {}
tools=CandidateTools([Turbine(**t) for t in data['turbines']],DeploymentConfig(**data['config']),
                     observer=progress,**data['limits'],**extra)
facade=ModuleType('candidate_tools')
def bind_tool(name):
    method=getattr(tools,name)
    if contract_version<2: return method
    signature=inspect.signature(method)
    def invoke(*args,**kwargs):
        try: signature.bind(*args,**kwargs)
        except TypeError:
            tools._error(name,'arguments',str(signature),kwargs,'invalid_arguments')
        result=method(*args,**kwargs)
        if contract_version>=3: tools._record_external_call(name)
        return result
    return invoke
for name in PUBLIC_METHODS: setattr(facade,name,bind_tool(name))
facade.BudgetExceeded=BudgetExceeded
if contract_version>=2: facade.ToolContractError=tool_runtime.ToolContractError
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
failure=None
answer=None
try:
    exec(compile(data['source'],sys.argv[1],'exec'),scope,scope)
    answer=scope['solve']([TurbineInput(t) for t in data['turbines']],C,int(data['seed']))
except BudgetExceeded as error:
    if not tools.stats().get('contract_error_count',0): answer=tools.best()
    if answer is None: failure=RuntimeError('Budget exhausted without explicitly remembered feasible deployment')
    termination='budget_exhausted'
except Exception as error:
    failure=error
if failure is None:
    if contract_version>=2:
        try: tools._proposal_schema('solve',answer)
        except tool_runtime.ToolContractError as error: failure=error
    elif not isinstance(answer,dict) or set(answer)!={'nest_ids','assignments'}:
        failure=ValueError('solve must return exactly nest_ids and assignments')
telemetry=dict(tools.stats(),termination=termination)
if telemetry.get('contract_error_count',0):
    telemetry['termination']='tool_contract_error'
    json.dump({'status':'tool_contract_error','proposal':None,'telemetry':telemetry,
               'final_error_type':type(failure).__name__ if failure else None},sys.stdout,allow_nan=False)
elif failure is not None:
    raise failure
else:
    json.dump({'status':'returned','proposal':answer,'telemetry':telemetry},sys.stdout,allow_nan=False)
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
    contract_events: list[dict] = []
    contract_event_count = 0
    timed_out = False
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

    def handle_event(row):
        nonlocal contract_event_count
        if row.get("kind") == "tool_contract_error":
            contract_event_count += 1
            if len(contract_events) < 5:
                contract_events.append({k: v for k, v in row.items() if k != "kind"})
            if telemetry is not None:
                telemetry.update(
                    contract_error_count=contract_event_count,
                    contract_errors=list(contract_events),
                )
        if observer:
            observer(row)

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
                handle_event(row)
            remaining = timeout_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                if telemetry is not None:
                    telemetry.update(
                        termination="hard_timeout",
                        seconds=time.perf_counter() - started,
                    )
                timed_out = True
                break
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
            handle_event(row)
    if timed_out:
        if contract_events:
            details = dict(
                telemetry or {},
                contract_errors=contract_events,
                contract_error_count=contract_event_count,
                contract_error_count_is_lower_bound=True,
                termination="hard_timeout",
                final_error_type="TimeoutError",
            )
            if telemetry is not None:
                telemetry.update(details)
            raise CandidateExecutionError(
                contract_events[0]["message"], telemetry=details
            )
        raise TimeoutError(f"Candidate exceeded {timeout_seconds}s time limit.")
    if proc.returncode:
        if contract_events:
            details = dict(
                telemetry or {},
                contract_errors=contract_events,
                contract_error_count=contract_event_count,
                contract_error_count_is_lower_bound=True,
                termination="tool_contract_error",
            )
            if telemetry is not None:
                telemetry.update(details)
            raise CandidateExecutionError(
                contract_events[0]["message"], telemetry=details
            )
        raise RuntimeError(
            ("".join(errors) or f"candidate exited {proc.returncode}")[-3000:]
        )
    try:
        envelope = json.loads("".join(output))
        result = envelope["proposal"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Candidate returned invalid JSON.") from error
    details = envelope.get("telemetry", {})
    if envelope.get("final_error_type"):
        details["final_error_type"] = envelope["final_error_type"]
    if telemetry is not None:
        telemetry.update(details)
    if envelope.get("status") == "tool_contract_error":
        raise CandidateExecutionError(
            details["contract_errors"][0]["message"], telemetry=details
        )
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("nest_ids"), list)
        or not isinstance(result.get("assignments"), dict)
    ):
        raise ValueError(
            "Candidate output must contain list nest_ids and mapping assignments."
        )
    if any(not isinstance(n, str) for n in result["nest_ids"]) or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in result["assignments"].items()
    ):
        raise ValueError(
            "Candidate IDs must be strings; automatic str() coercion is forbidden."
        )
    return CandidateProposal(
        list(result["nest_ids"]),
        dict(result["assignments"]),
    )
