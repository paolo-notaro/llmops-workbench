FROM python:3.11-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.2.1

COPY pyproject.toml poetry.lock README.md LICENSE ./
COPY llmops_workbench ./llmops_workbench

RUN poetry install --only main --no-interaction

FROM python:3.11-slim

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY llmops_workbench ./llmops_workbench
COPY examples ./examples
COPY datasets ./datasets
COPY docs ./docs
COPY frontend ./frontend

RUN useradd --create-home --uid 10001 appuser

USER appuser
EXPOSE 8000

CMD ["uvicorn", "llmops_workbench.app:app", "--host", "0.0.0.0", "--port", "8000"]
