"""Separate human progress, LLM context and content-addressed diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def redact_secrets(value, secrets=()):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(s in k.lower() for s in
                ("api_key", "secret", "password", "authorization")) else
                redact_secrets(v, secrets)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(v, secrets) for v in value]
    if isinstance(value, str):
        value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False,
                                indent=2, sort_keys=True) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("run-id only accepts ASCII letters, digits, '-' and '_'.")
    return value


class Records:
    def __init__(self, root, secrets=()):
        self.root = Path(root)
        self.secrets = secrets
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def json(self, relative, value):
        atomic_json(self.root / relative, redact_secrets(value, self.secrets))

    def text(self, relative, value):
        atomic_text(self.root / relative, redact_secrets(value, self.secrets))

    def event(self, event, **fields):
        self.append("diagnostics/events.jsonl", {"event": event, **fields})

    def append(self, relative, fields):
        row = redact_secrets({"time": datetime.now(timezone.utc).isoformat(),
                              **fields}, self.secrets)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def blob(self, value):
        key = signature(value)
        path = "diagnostics/objects/" + key + ".json"
        with self.lock:
            if not (self.root / path).exists():
                self.json(path, value)
        return path

    def progress(self, message):
        safe = str(redact_secrets(message, self.secrets))
        safe = "".join(c for c in safe if unicodedata.category(c) not in ("Cc", "Cf"))
        line = datetime.now().strftime("%H:%M:%S") + " " + safe
        with self.lock, (self.root / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            print(line, flush=True)
        self.event("progress", message=safe)
