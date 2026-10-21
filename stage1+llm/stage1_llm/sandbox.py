"""Best-effort AST-restricted execution in a timed isolated Python interpreter."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from .candidate import CandidateProposal, validate_candidate_source
from .domain import DeploymentConfig, Turbine

_WORKER = r'''import json,math,random,itertools,collections,typing,sys
from types import SimpleNamespace
allowed={"abs":abs,"all":all,"any":any,"bool":bool,"dict":dict,"enumerate":enumerate,"filter":filter,"float":float,"frozenset":frozenset,"int":int,"isinstance":isinstance,"len":len,"list":list,"map":map,"max":max,"min":min,"next":next,"range":range,"reversed":reversed,"round":round,"set":set,"sorted":sorted,"str":str,"sum":sum,"tuple":tuple,"zip":zip,"Exception":Exception,"ValueError":ValueError,"KeyError":KeyError}
mods={"math":math,"random":random,"itertools":itertools,"collections":collections,"typing":typing}
def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
    if name not in mods: raise ImportError("module not allowed")
    return mods[name]
allowed["__import__"]=safe_import
data=json.load(sys.stdin); scope={"__builtins__":allowed,"math":math,"random":random}
exec(compile(data["source"],sys.argv[1],"exec"),scope,scope)
T=lambda row: SimpleNamespace(identifier=row["identifier"],x_km=row["x_km"],y_km=row["y_km"])
C=SimpleNamespace(**data["config"])
answer=scope["solve"]([T(t) for t in data["turbines"]],C,int(data["seed"]))
if not isinstance(answer,dict) or set(answer)!={"nest_ids","assignments"}: raise ValueError("solve must return exactly nest_ids and assignments")
json.dump(answer,sys.stdout,allow_nan=False)
'''

def run_candidate(source_path: Path, turbines: list[Turbine], config: DeploymentConfig, seed: int, timeout_seconds: int) -> CandidateProposal:
    source = source_path.read_text(encoding="utf-8")
    validate_candidate_source(source)
    payload = {"turbines":[{"identifier":t.identifier,"x_km":t.x_km,"y_km":t.y_km} for t in turbines], "config":{n:getattr(config,n) for n in config.__dataclass_fields__}, "seed":seed}
    payload['source'] = source
    try: proc = subprocess.run([sys.executable,"-I","-X","utf8","-c",_WORKER,str(source_path)], input=json.dumps(payload), text=True, encoding='utf-8', capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error: raise TimeoutError(f"Candidate exceeded {timeout_seconds}s time limit.") from error
    if proc.returncode: raise RuntimeError((proc.stderr or f"candidate exited {proc.returncode}")[-3000:])
    try: result=json.loads(proc.stdout)
    except json.JSONDecodeError as error: raise ValueError("Candidate returned invalid JSON.") from error
    if not isinstance(result.get("nest_ids"),list) or not isinstance(result.get("assignments"),dict): raise ValueError("Candidate output must contain list nest_ids and mapping assignments.")
    return CandidateProposal([str(x) for x in result["nest_ids"]], {str(k):str(v) for k,v in result["assignments"].items()})
