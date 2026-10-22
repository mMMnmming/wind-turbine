"""Frozen Stage2 greedy oracle; trusted worker, actual-route audit and caching."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import logging
import math
import multiprocessing
from pathlib import Path
import sys
import time

from .domain import DeploymentConfig, Turbine
from .sandbox import CandidateDeadlineExceeded

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "stage2/outputs/finetune-edge-v2/control/best_greedy.pt"
)


class InfrastructureError(RuntimeError):
    """A broken oracle must never be classified as a poor deployment."""


def environment(config: DeploymentConfig, gust: float) -> dict:
    """Use exactly the environment fields consumed by Stage2."""
    from stage2.atsp import WindATSPConfig

    values = asdict(config)
    values["gust_speed_mps"] = gust
    result = {k: values[k] for k in WindATSPConfig.__dataclass_fields__}
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in result.values()
    ):
        raise ValueError("Stage2 environment must contain finite numeric values.")
    if gust < 0:
        raise ValueError("gust_speed_mps must be nonnegative.")
    return result


def stage2_fingerprint() -> str:
    """Version all local Stage2 Python sources, without editing or executing training."""
    digest = hashlib.sha256()
    for path in sorted((REPO_ROOT / "stage2").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


class GreedyDecoder:
    """Load a user-trusted checkpoint once. Never fall back to another policy."""

    def __init__(
        self, checkpoint: Path, device: str, config: DeploymentConfig, gust: float
    ):
        import torch
        from stage2.policy import build_policy
        from .artifacts import sha256_file

        if device not in ("cpu", "cuda"):
            raise ValueError("stage2_device must be cpu or cuda.")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage2 checkpoint not found: {checkpoint}")
        if device == "cuda" and not torch.cuda.is_available():
            raise InfrastructureError(
                "CUDA unavailable; explicitly select --stage2-device cpu if intended."
            )
        # Only the operator-supplied checkpoint is deserialized, never candidate data.
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        training = payload["training_config"]
        architecture = payload.get(
            "architecture", training.get("architecture", "legacy")
        )
        if architecture != training.get("architecture", "legacy"):
            raise ValueError("Checkpoint architecture metadata is inconsistent.")
        actual_environment = environment(config, gust)
        trained = payload.get("wind_environment")
        if trained:
            from stage2.atsp import WindATSPConfig

            if asdict(WindATSPConfig(**trained)) != actual_environment:
                raise ValueError(
                    "Checkpoint environment differs from the explicit deployment wind/endurance environment."
                )
        else:
            logging.warning(
                "Checkpoint lacks wind_environment; explicit deployment environment remains authoritative."
            )
        self.device = torch.device(device)
        self.model = build_policy(training).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.metadata = {
            "origin": "frozen_stage2",
            "checkpoint_sha256": sha256_file(checkpoint),
            "architecture": architecture,
            "device": device,
            "decode_strategy": "greedy",
            "candidate_count": 1,
            "environment": actual_environment,
            "training_environment": trained,
            "stage2_fingerprint": stage2_fingerprint(),
        }

    def decode(self, instance):
        from stage2.policy import infer_instance

        result = infer_instance(
            self.model, instance, self.device, strategy="greedy", candidates=1
        )
        return result["route"], result["runtime_seconds"]


class CheckedPlanner:
    """Canonical geometry + Stage2 route checks; no heuristic route or NN audit."""

    def __init__(self, decoder, config: DeploymentConfig, gust: float = 12.0):
        self.decoder = decoder
        self.config = config
        self.env = environment(config, gust)
        self.metadata = {**decoder.metadata, "environment": self.env}
        self.cache: dict[str, dict] = {}
        self.requests = self.inferences = self.hits = 0

    def region(self, points: list[Turbine], nest: str) -> dict:
        import numpy as np
        from stage2.atsp import WindATSPConfig, WindATSPInstance

        self.requests += 1
        by_id = {p.identifier: p for p in points}
        if not points or len(by_id) != len(points) or nest not in by_id:
            raise ValueError("Oracle region requires unique points and a member depot.")
        ordered = [by_id[nest], *(by_id[k] for k in sorted(by_id) if k != nest)]
        coordinates = np.asarray([(p.x_km, p.y_km) for p in ordered], dtype=float)
        instance = WindATSPInstance(
            tuple(p.identifier for p in ordered),
            coordinates,
            0,
            WindATSPConfig(**self.env),
        )
        # Validate wind/nonpositive ground speed even for singleton/cheap rejections.
        _ = instance.travel_time_seconds
        key = hashlib.sha256(
            json.dumps(
                {
                    "points": [asdict(p) for p in ordered],
                    "oracle": self.metadata,
                    "service_radius_km": self.config.communication_radius_km,
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if key in self.cache:
            self.hits += 1
            return {**copy.deepcopy(self.cache[key]), "cache_hit": True}
        radius = float(np.linalg.norm(coordinates - coordinates[0], axis=1).max())
        if radius > self.config.communication_radius_km + 1e-9:
            return {
                "feasible": False,
                "diagnostics": [
                    {
                        "kind": "service_radius",
                        "actual": radius,
                        "limit": self.config.communication_radius_km,
                        "excess": radius - self.config.communication_radius_km,
                    }
                ],
                "task_count": len(points) - 1,
                "radius_km": radius,
                "cache_hit": False,
            }
        if len(points) == 1:
            route, runtime = (0, 0), 0.0
        else:
            try:
                route, runtime = self.decoder.decode(instance)
            except Exception as error:
                raise InfrastructureError(
                    f"Stage2 decoder failed: {type(error).__name__}: {error}"
                ) from error
            self.inferences += 1
        try:
            metrics = instance.evaluate_route(tuple(route))
            if (
                not all(
                    math.isfinite(v)
                    for v in (
                        metrics.flight_seconds,
                        metrics.equivalent_energy_seconds,
                        runtime,
                    )
                )
                or runtime < 0
            ):
                raise ValueError("Nonfinite or negative timing returned by Stage2.")
        except Exception as error:
            raise InfrastructureError(f"Invalid Stage2 output: {error}") from error
        margin = self.config.max_endurance_seconds - metrics.equivalent_energy_seconds
        diagnostics = (
            []
            if metrics.feasible
            else [
                {
                    "kind": "network_route_endurance",
                    "actual": metrics.equivalent_energy_seconds,
                    "limit": self.config.max_endurance_seconds,
                    "excess": -margin,
                    "normalized_excess": -margin / self.config.max_endurance_seconds,
                }
            ]
        )
        result = {
            "feasible": metrics.feasible,
            "diagnostics": diagnostics,
            "nest_id": nest,
            "task_count": len(points) - 1,
            "radius_km": radius,
            "radius_margin_km": self.config.communication_radius_km - radius,
            "flight_seconds": metrics.flight_seconds,
            "energy_seconds": metrics.equivalent_energy_seconds,
            "endurance_margin_seconds": margin,
            "route_ids": [instance.identifiers[i] for i in route],
            "runtime_seconds": runtime,
            "cache_key": key,
        }
        self.cache[key] = copy.deepcopy(result)
        return {**result, "cache_hit": False}

    def stats(self) -> dict:
        return {
            "requests": self.requests,
            "inferences": self.inferences,
            "cache_hits": self.hits,
        }


def _serve(connection, checkpoint: str, device: str, config: dict, gust: float):
    """Only trusted parent communicates over this private multiprocessing pipe."""
    try:
        decoder = GreedyDecoder(
            Path(checkpoint), device, DeploymentConfig(**config), gust
        )
        planner = CheckedPlanner(decoder, DeploymentConfig(**config), gust)
        connection.send({"ok": True, "metadata": planner.metadata})
        while True:
            request = connection.recv()
            if request is None:
                break
            try:
                result = planner.region(
                    [Turbine(**p) for p in request["points"]], request["nest"]
                )
                connection.send(
                    {"ok": True, "result": result, "stats": planner.stats()}
                )
            except Exception as error:
                connection.send(
                    {"ok": False, "error": f"{type(error).__name__}: {error}"}
                )
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


class PlannerService:
    """Killable persistent worker, so CUDA faults never hang the search supervisor."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        config: DeploymentConfig,
        gust=12.0,
        timeout=300.0,
    ):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_serve,
            args=(child, str(checkpoint), device, asdict(config), gust),
            daemon=True,
        )
        self.timeout = timeout
        self.counters = {}
        self._pending_started = None
        self.process.start()
        child.close()
        try:
            self.metadata = self._receive(timeout)["metadata"]
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout, *, candidate_deadline=False):
        try:
            deadline = time.perf_counter() + max(0.0, timeout)
            while not self.connection.poll(
                min(10.0, max(0.0, deadline - time.perf_counter()))
            ):
                if time.perf_counter() >= deadline:
                    if candidate_deadline:
                        raise CandidateDeadlineExceeded(
                            "Candidate deadline reached during Stage2 inference."
                        )
                    raise InfrastructureError(
                        "Stage2 worker timeout; no automatic restart or policy fallback."
                    )
                print(
                    "[stage2_heartbeat] Trusted greedy inference worker still active",
                    flush=True,
                )
            response = self.connection.recv()
            self._pending_started = None
            if not response["ok"]:
                raise InfrastructureError(response["error"])
            return response
        except CandidateDeadlineExceeded:
            raise
        except (EOFError, BrokenPipeError, OSError) as error:
            raise InfrastructureError("Stage2 worker disconnected.") from error

    def region(self, points, nest, timeout=None):
        if self._pending_started is not None:
            self.settle()
        try:
            self.connection.send({"points": [asdict(p) for p in points], "nest": nest})
            self._pending_started = time.perf_counter()
        except (OSError, EOFError) as error:
            raise InfrastructureError("Stage2 worker unavailable.") from error
        response = self._receive(
            self.timeout if timeout is None else min(timeout, self.timeout),
            candidate_deadline=timeout is not None and timeout < self.timeout,
        )
        self.counters = response["stats"]
        return response["result"]

    def settle(self):
        """Drain a timed-out candidate response before any subsequent request/API round.

        Candidate execution is already killed. A genuinely stuck worker still fails
        at its original infrastructure deadline; no reply can leak into a new request.
        """
        if self._pending_started is not None:
            response = self._receive(
                max(0.0, self.timeout - (time.perf_counter() - self._pending_started))
            )
            self.counters = response["stats"]

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        self.connection.close()

    def stats(self):
        return dict(self.counters)
