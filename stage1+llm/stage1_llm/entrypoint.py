"""Dispatch legacy artifacts/configurations without silently changing semantics."""

import json
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = False
    if "--search-config" in argv:
        index = argv.index("--search-config")
        if index + 1 < len(argv):
            path = Path(argv[index + 1])
            if path.is_file():
                legacy = (
                    json.loads(path.read_text(encoding="utf-8")).get("search_version")
                    != 5
                )
    if (
        argv
        and argv[0] in ("list-candidates", "export-best", "resume")
        and "--search-config" not in argv
    ):
        option = "--resume-run-id" if argv[0] == "resume" else "--run-id"
        if option in argv and argv.index(option) + 1 < len(argv):
            root = Path(__file__).resolve().parents[1] / "outputs"
            if "--outputs" in argv:
                root = Path(argv[argv.index("--outputs") + 1])
            from .artifacts import run_path

            state_path = (
                run_path(root, argv[argv.index(option) + 1]) / "search_state.json"
            )
            if state_path.is_file():
                legacy = (
                    json.loads(state_path.read_text(encoding="utf-8")).get(
                        "search_version"
                    )
                    != 5
                )
    if (
        argv
        and argv[0] == "evaluate"
        and "--search-config" not in argv
        and "--candidate" in argv
    ):
        index = argv.index("--candidate")
        if index + 1 < len(argv):
            selection = Path(argv[index + 1]).parent / "selection.json"
            if selection.is_file():
                legacy = (
                    json.loads(selection.read_text(encoding="utf-8")).get(
                        "search_version"
                    )
                    != 5
                )
    if legacy:
        from .cli import main as implementation
    else:
        from .joint_cli import main as implementation
    return implementation(argv)
