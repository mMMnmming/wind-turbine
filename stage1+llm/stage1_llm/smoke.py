"""Offline simulated end-to-end check; never constructs a paid API client."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .client import DeepSeekSettings, ModelCandidate
from .artifacts import run_path, validate_run_id
from .domain import DeploymentConfig, load_turbines
from .evaluator import evaluate_proposal
from .exports import export_candidate, export_evaluation, list_candidates
from .sandbox import run_candidate
from .search import SearchController
from .settings import REPO_ROOT, SearchSettings

SOLO = "def solve(turbines, config, seed):\n return {'nest_ids':[t.identifier for t in turbines], 'assignments':{t.identifier:t.identifier for t in turbines}}\n"
GROUP = "def solve(turbines, config, seed):\n n=turbines[0].identifier\n return {'nest_ids':[n], 'assignments':{t.identifier:n for t in turbines}}\n"
BROKEN = "def solve(turbines, config, seed):\n return {'nest_ids':[turbines[0].identifier], 'assignments':{}}\n"
TOOL_GROUP = """from candidate_tools import split_repair, remember, make_proposal, assignments_from_proposal
def solve(turbines, config, seed):
    initial = make_proposal({t.identifier:turbines[0].identifier for t in turbines})
    answer = split_repair(assignments_from_proposal(initial))
    remember(answer)
    return answer
"""


class MockClient:
    def __init__(self, sources: list[str | Exception]):
        self.sources = iter(sources)

    def generate(self, _system: str, _prompt: str, direction: str) -> ModelCandidate:
        source = next(self.sources)
        if isinstance(source, Exception):
            raise source
        return ModelCandidate(
            source,
            "离线模拟，不代表研究算法",
            "检查恢复与修复",
            "验证软件流程",
            "mock fixture only",
            direction,
        )


def smoke_test(outputs: Path, run_id: str) -> Path:
    validate_run_id(run_id)
    fixture_root = run_path(outputs, run_id + "-fixtures")
    fixture_root.mkdir(parents=True, exist_ok=False)
    source = fixture_root / "tiny.tsv"
    source.write_text("a\t0\t0\nb\t.1\t0\nc\t0\t.1\nd\t.1\t.1\n", encoding="utf-8")
    settings = SearchSettings(
        source,
        (),
        DeploymentConfig(),
        data_dir=fixture_root / "data",
        protected_input=REPO_ROOT / "turbine87.tsv",
        training_virtual_fields=1,
        validation_virtual_fields=1,
        initial_candidates=1,
        evaluation_budget=2,
        selector="sliding_window",
        constraint_preflight=True,
        early_rejection=True,
    )
    llm = DeepSeekSettings("https://example.invalid", "offline_mock", "not-a-real-key")
    first = SearchController(
        settings, llm, outputs, client=MockClient([BROKEN, TOOL_GROUP])
    )
    path = first.run(run_id)
    # Resume, introduce a genuine tradeoff, duplicate, and generation failure.
    resumed = SearchController(
        replace(settings, evaluation_budget=5),
        llm,
        outputs,
        client=MockClient(
            [
                SOLO,
                TOOL_GROUP + "\n# same behavior",
                ValueError("simulated API failure"),
            ]
        ),
    )
    resumed.run(run_id, resume=True)
    rows = list_candidates(path)
    assert len(rows) == 2 and len(resumed.history) == 5
    exported = export_candidate(
        path,
        rows[0]["candidate_id"],
        allow_mock=True,
        reason="Offline smoke validation only.",
    )
    points = load_turbines(source)
    checked = evaluate_proposal(
        points,
        settings.deployment,
        run_candidate(
            exported / "algorithm.py", points, settings.deployment, settings.seed, 10
        ),
    )
    export_evaluation(
        path / "mock-deployment",
        points,
        checked,
        {"kind": "offline_mock", "not_formal_result": True},
    )
    print(
        f"OFFLINE SMOKE PASS: {path}; mock artifacts are not formal algorithms or experiment results.",
        flush=True,
    )
    return path
