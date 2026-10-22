"""Offline six-round search/repair/resume/export journey. No trained-model claims."""

from dataclasses import replace
import json
from pathlib import Path

from .artifacts import atomic_text, run_path
from .candidate import ModelCandidate
from .client import DeepSeekSettings
from .domain import DeploymentConfig
from .joint_artifacts import export_joint_candidate, export_deployment
from .joint_planner import CheckedPlanner
from .joint_search import JointSearch
from .joint_settings import JointSettings
from .data import turbines_from_record
from .settings import SearchSettings


class MockDecoder:
    """Fixed input-order tour, not trained and not a nearest-neighbour route."""

    metadata = {
        "origin": "offline_mock",
        "checkpoint_sha256": "offline-mock-only",
        "architecture": "offline_mock",
        "device": "cpu",
        "decode_strategy": "greedy",
        "candidate_count": 1,
    }

    def decode(self, instance):
        return tuple(range(instance.node_count)) + (0,), 0.0


class MockClient:
    def __init__(self, sources):
        self.sources = iter(sources)
        self.calls = 0

    def generate(self, system, prompt, direction):
        self.calls += 1
        source = next(self.sources)
        if isinstance(source, Exception):
            raise source
        return ModelCandidate(
            source,
            "离线模拟算法",
            "离线模拟修改",
            "无真实性能声明",
            "离线测试",
            direction,
        )

    def close(self):
        pass


SOLO = "def solve(turbines, config, seed):\n return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}\n"
GROUP = "from candidate_tools import repartition_repair,remember\ndef solve(turbines, config, seed):\n p=repartition_repair({t.identifier:turbines[0].identifier for t in turbines})\n remember(p)\n return p\n"
BROKEN = "from candidate_tools import region\ndef solve(turbines, config, seed):\n region((0,0),[])\n return {}\n"


def smoke_test(outputs: Path, run_id=None):
    from .joint_planner import REPO_ROOT

    run_id = run_id or "offline-joint-v5"
    if run_path(outputs, run_id).exists():
        raise FileExistsError("Smoke run exists; choose a new --run-id.")
    fixture = outputs / (run_id + "-fixture")
    fixture.mkdir(parents=True, exist_ok=False)
    atomic_text(fixture / "tiny.tsv", "a\t0\t0\nb\t0.1\t0\nc\t0\t0.1\nd\t0.1\t0.1\n")
    settings = SearchSettings(
        fixture / "tiny.tsv",
        (),
        DeploymentConfig(),
        data_dir=fixture / "data",
        protected_input=REPO_ROOT / "turbine87.tsv",
        training_virtual_fields=0,
        validation_virtual_fields=0,
        evaluation_budget=3,
        no_improvement_limit=10,
        initial_candidates=3,
        show_code=False,
        heartbeat_seconds=10.0,
    )
    joint = JointSettings(settings, device="cpu")
    llm = DeepSeekSettings("https://example.invalid", "offline_mock", "")
    planner = CheckedPlanner(MockDecoder(), settings.deployment)
    client = MockClient([BROKEN, SOLO, GROUP])
    first = JointSearch(joint, llm, outputs, planner, client)
    root = first.run(run_id)
    second = JointSearch(
        replace(joint, search=replace(settings, evaluation_budget=6)),
        llm,
        outputs,
        planner,
        MockClient(
            [
                GROUP,
                ValueError("simulated provider schema failure"),
                GROUP + "\n# distinct source, duplicate behavior\n",
            ]
        ),
    )
    second.run(run_id, resume=True)
    state = json.loads((root / "search_state.json").read_text(encoding="utf-8"))
    assert len(state["history"]) == 6 and state["archive"]
    export_joint_candidate(root, allow_mock=True)
    best = state["archive"][0]
    dataset = best["evaluation"]["datasets"][0]
    from .candidate import CandidateProposal

    export_deployment(
        root / "mock-deployment",
        turbines_from_record(first.bundle["datasets"][0]),
        CandidateProposal(**dataset["proposal"]),
        dataset,
        planner.metadata,
        settings.deployment,
        plots=False,
    )
    print(
        f"OFFLINE MOCK ONLY: repair, search, six rounds, resume and export passed: {root}",
        flush=True,
    )
    return root
