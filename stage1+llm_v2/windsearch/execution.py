"""Killable JSON protocol with independent progress heartbeats."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .planner import InfrastructureError


class StrategyError(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail.get("error", "Strategy failed"))


class StrategyCallbackTimeout(TimeoutError):
    def __init__(self, operation, elapsed, limit):
        self.detail = {"operation": operation, "elapsed_seconds": elapsed,
                       "limit_seconds": limit, "reason": "callback_watchdog"}
        super().__init__(f"{operation} did not return within {limit:g} seconds")


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
                    self.queue.put((time.perf_counter(), {"kind": "protocol_error", "error": "Worker output exceeds protocol limit"}))
                    break
                try:
                    self.queue.put((time.perf_counter(), json.loads(line)))
                except ValueError:
                    self.queue.put((time.perf_counter(), {"kind": "protocol_error", "error": "Invalid worker JSON"}))
                    break
        finally:
            self.queue.put((time.perf_counter(), {"kind": "protocol_error", "error": "Strategy worker disconnected"}))

    def _stderr(self):
        for line in self.process.stderr:
            self.records.append("diagnostics/errors/worker-stderr.jsonl",
                                {"scope": self.prefix, "text": line.rstrip()})

    def exchange(self, payload, timeout_seconds, heartbeat, pulse, *, operation):
        self.calls += 1
        request_ref = self.records.blob(payload)
        self.records.append("diagnostics/tool_calls/" + self.prefix + ".jsonl",
                            {"call": self.calls, "direction": "input", "object": request_ref})
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
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
        dispatched = time.perf_counter()
        # Startup/transport hangs are infrastructure faults, not slow strategy code.
        transport_deadline = dispatched + max(30.0, timeout_seconds)
        callback_started = None
        next_pulse = dispatched + heartbeat
        while True:
            if sent.is_set() and send_errors:
                raise InfrastructureError("Strategy transport failed: " + send_errors[0])
            deadline = (callback_started + timeout_seconds if callback_started is not None
                        else transport_deadline)
            try:
                # Drain already received output BEFORE checking the supervisor clock.
                # Slow diagnostic writes must not turn an on-time reply into a timeout.
                received, row = self.queue.get_nowait()
            except queue.Empty:
                now = time.perf_counter()
                if now >= deadline:
                    if callback_started is None:
                        raise InfrastructureError("Strategy worker startup/transport timeout before callback start")
                    raise StrategyCallbackTimeout(operation, now-callback_started, timeout_seconds)
                try:
                    received, row = self.queue.get(timeout=min(.25, deadline-now))
                except queue.Empty:
                    if time.perf_counter() >= next_pulse:
                        pulse()
                        next_pulse = time.perf_counter() + heartbeat
                    continue
            if not isinstance(row, dict) or row.get("kind") == "protocol_error":
                raise InfrastructureError("Strategy protocol failed: " + str(row))
            if row.get("kind") == "callback_started":
                if callback_started is not None or row.get("operation") != operation:
                    raise InfrastructureError("Unexpected strategy callback start")
                callback_started = row["started_at"]
                self.records.event("callback_started", scope=self.prefix, operation=operation,
                                   call=self.calls, timeout_seconds=timeout_seconds)
                continue
            if callback_started is None:
                raise InfrastructureError("Strategy response arrived before callback start")
            if received > callback_started + timeout_seconds:
                raise StrategyCallbackTimeout(operation, received-callback_started, timeout_seconds)
            ref = self.records.blob(row)
            self.records.append("diagnostics/tool_calls/" + self.prefix + ".jsonl",
                                {"call": self.calls, "direction": "output", "object": ref,
                                 "callback_wall_seconds": received-callback_started,
                                 "exchange_wall_seconds": time.perf_counter()-dispatched})
            if not row.get("ok"):
                if row.get("kind") != "strategy_error":
                    raise InfrastructureError("Unexpected strategy failure response")
                raise StrategyError(row)
            if row.get("kind") != ("initialized" if operation == "initialize" else "proposal"):
                raise InfrastructureError("Unexpected strategy response kind")
            return row

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=2)
        self.stderr.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
