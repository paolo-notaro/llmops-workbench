"""Structured parsing for the controlled sanitized case-study format."""

from __future__ import annotations

import json
from pathlib import Path

META = {
    "01-llm-evaluation-platform": {"slug": "llm-evaluation-platform", "themes": ["evaluation", "CI gates", "promotion"]},
    "02-rag-observability-stack": {"slug": "rag-observability-stack", "themes": ["retrieval", "citations", "telemetry"]},
    "03-vllm-kubernetes-serving": {"slug": "vllm-kubernetes-serving", "themes": ["serving", "Kubernetes", "rollout"]},
    "04-ml-security-privacy": {"slug": "ml-security-privacy", "themes": ["guardrails", "privacy", "threats"]},
    "05-aiops-reliability": {"slug": "aiops-reliability", "themes": ["reliability", "incidents", "observability"]},
}


def parse_case_study(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    sections: list[dict[str, object]] = []
    heading = ""
    paragraph: list[str] = []
    list_items: list[str] = []
    blocks: list[dict[str, object]] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append({"type": "paragraph", "text": " ".join(paragraph)})
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            blocks.append({"type": "list", "items": list(list_items)})
            list_items.clear()

    def flush_section() -> None:
        nonlocal blocks
        flush_paragraph()
        flush_list()
        if heading:
            sections.append({"heading": heading, "blocks": blocks})
        blocks = []

    for line in lines[1:]:
        if line.startswith("## "):
            flush_section()
            heading = line.removeprefix("## ").strip()
        elif line.startswith("- "):
            flush_paragraph()
            list_items.append(line.removeprefix("- ").strip())
        elif not line.strip():
            flush_paragraph()
            flush_list()
        else:
            paragraph.append(line.strip())
    flush_section()
    meta = META[path.stem]
    problem = next(section for section in sections if section["heading"] == "Problem")
    summary = next(block["text"] for block in problem["blocks"] if block["type"] == "paragraph")
    return {
        "number": path.stem.split("-", 1)[0], "slug": meta["slug"], "title": title,
        "themes": meta["themes"], "summary": summary, "sections": sections,
    }


def build_case_studies(source_dir: Path, output_path: Path) -> list[dict[str, object]]:
    """Build and write structured case-study data from canonical Markdown."""

    studies = [parse_case_study(path) for path in sorted(source_dir.glob("*.md"))]
    output_path.write_text(json.dumps({"studies": studies}, indent=2) + "\n", encoding="utf-8")
    return studies
