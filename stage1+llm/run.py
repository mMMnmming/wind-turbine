"""CLI entry point for the independent Stage 1+LLM search package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_llm.entrypoint import main

if __name__ == "__main__":
    raise SystemExit(main())
