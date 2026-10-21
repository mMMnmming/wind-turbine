"""Run-scoped artifact persistence, hashing, and secret-safe logs."""
from __future__ import annotations
import hashlib, json, re
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',value):
        raise ValueError("run_id may contain only ASCII letters, digits, '_' and '-'.")
    return value


def safe_child(root: Path, relative: str) -> Path:
    result = (root/relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Output path escapes its configured root.')
    return result


def run_path(root: Path, run_id: str) -> Path:
    return safe_child(root,validate_run_id(run_id))

def redact_secrets(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(word in k.lower() for word in ("api_key", "secret", "password")) or k.lower() in ('token','authorization','access_token','refresh_token') else redact_secrets(v, secrets)) for k,v in value.items()}
    if isinstance(value, list): return [redact_secrets(v, secrets) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(sk-[A-Za-z0-9_-]{8,})", "[REDACTED]", value)
        for secret in secrets:
            if secret: value = value.replace(secret, "[REDACTED]")
        return value
    return value

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()

class RunArtifacts:
    def __init__(self, root: Path, run_id: str | None = None, secrets: tuple[str, ...] = ()):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
        self.secrets = secrets
        self.root = run_path(root,self.run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "candidates").mkdir(exist_ok=True); (self.root / "evaluations").mkdir(exist_ok=True)
        self.log_path = self.root / "search.log"
        self._lock = threading.RLock()
    def write_json(self, relative: str, value: Any) -> None:
        atomic_json(self.root / relative, redact_secrets(value, self.secrets))
    def write_text(self, relative: str, value: str) -> None:
        atomic_text(safe_child(self.root,relative),value)
    def log(self, message: str, filename: str = "search.log", *, event: str = 'message', **fields: Any) -> None:
        safe = str(redact_secrets(message, self.secrets))
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        line = f'{timestamp} [{event}] {safe}'
        record = redact_secrets({'timestamp':timestamp, 'event':event, 'message':safe, **fields},self.secrets)
        with self._lock:
            with (self.root/filename).open('a',encoding='utf-8') as handle:
                handle.write(line+'\n')
            with (self.root/'events.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            print(line,flush=True)

    @contextmanager
    def heartbeat(self, label: str, interval: float = 10.0):
        stop = threading.Event()
        start = time.perf_counter()
        def pulse():
            while not stop.wait(interval):
                self.log(f'{label}; elapsed={time.perf_counter()-start:.1f}s',event='heartbeat')
        thread = threading.Thread(target=pulse,daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


def atomic_json(path: Path, value: Any) -> None:
    """Never leave a truncated state after interruption or serialization failure."""
    atomic_text(path,json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()
