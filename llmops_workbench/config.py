"""Configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    llm_provider: str = "mock"
    log_level: str = "INFO"
    docs_dir: Path = Field(default=REPO_ROOT / "examples" / "synthetic_docs")
    eval_dir: Path = Field(default=REPO_ROOT / "datasets" / "ground_truth")
    reports_dir: Path = Field(default=REPO_ROOT / "reports")
    max_latency_ms: float = 1000.0


def load_settings() -> Settings:
    """Load settings from `.env` and process environment."""

    load_dotenv()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower() or "mock",
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
    )
