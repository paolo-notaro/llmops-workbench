"""Build structured frontend data from the sanitized case-study Markdown files."""

from __future__ import annotations

from pathlib import Path

from llmops_workbench.case_studies import build_case_studies

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "docs" / "case-studies"
OUTPUT_PATH = REPO_ROOT / "frontend" / "case-studies-data.json"


def main() -> None:
    studies = build_case_studies(SOURCE_DIR, OUTPUT_PATH)
    print(f"Wrote {len(studies)} studies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
