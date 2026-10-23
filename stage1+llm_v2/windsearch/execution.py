"""Killable JSON protocol with independent progress heartbeats."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


class StrategyError(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail.get("error", "Strategy failed"))


class StrategyWorker:
    def __init__(self, records, prefix):
        self.records, self.prefix = records, prefix
        self.queue = queue.Queue()
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(Path(__file__).with_name("worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.stderr = threading.Thread(target=self._stderr, daemon=True)
        self.stderr.start()
        self.calls = 0

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(4_000_002)
                if not line:
                    break
                if len(line.encode("utf-8")) > 4_000_001 or not line.endswith("\n"):
                    self.queue.put({"ok": False, "error": "Worker output exceeds protocol limit"})
                    break
                try:
                    self.queue.put(json.loads(line))
                except ValueError:
                    self.queue.put({"ok": False, "error": "Invalid worker JSON"})
                    break
        finally:
            self.queue.put({"ok": False, "error": "Strategy worker disconnected"})

    def _stderr(self):
        for line in self.process.stderr:
            self.records.append("diagnostics/errors/worker-stderr.jsonl",
                                {"scope": self.prefix, "text": line.rstrip()})

    def exchange(self, payload, deadline, heartbeat, pulse):
        self.calls += 1
        started = time.perf_counter()
        request_ref = self.records.blob(payload)
        self.records.append("diagnostics/tool_calls/" + self.prefix + ".jsonl",
                            {"call": self.calls, "direction": "input", "object": request_ref})
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
        # A strategy that stops reading must not block the supervisor's heartbeat.
        sent = threading.Event()
        send_errors = []

        def send():
            try:
                self.process.stdin.write(encoded)
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                send_errors.append(str(error))
            finally:
                sent.set()

        threading.Thread(target=send, daemon=True).start()
        next_pulse = time.perf_counter() + heartbeat
        while True:
            now = time.perf_counter()
            if now >= deadline:
                raise TimeoutError("search time budget exhausted while strategy executes")
            if sent.is_set() and send_errors:
                raise StrategyError({"error": send_errors[0]})
            try:
                row = self.queue.get(timeout=min(0.25, deadline - now))
                ref = self.records.blob(row)
                self.records.append("diagnostics/tool_calls/" + self.prefix + ".jsonl",
                                    {"call": self.calls, "direction": "output", "object": ref,
                                     "wall_seconds": time.perf_counter() - started})
                if not row.get("ok"):
                    raise StrategyError(row)
                return row
            except queue.Empty:
                if time.perf_counter() >= next_pulse:
                    pulse()
                    next_pulse = time.perf_counter() + heartbeat

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=2)
        self.stderr.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
