"""Independent v2 entry point; does not import the older stage1+llm package."""
from windsearch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
