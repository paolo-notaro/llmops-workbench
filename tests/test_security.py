import asyncio
import json
import re
from dataclasses import dataclass

from llmops_workbench.app import app, live_monitoring, query
from llmops_workbench.config import REPO_ROOT
from llmops_workbench.models import QueryRequest
from llmops_workbench.security import (
    MAX_REQUEST_BODY_BYTES,
    PublicDemoSecurityMiddleware,
    SlidingWindowLimiter,
    TELEMETRY_REQUEST_LIMIT,
)


@dataclass
class ASGIResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> object:
        return json.loads(self.content)


def request(method: str, path: str, body: bytes = b"", content_type: str | None = None) -> ASGIResponse:
    async def invoke() -> ASGIResponse:
        messages: list[dict[str, object]] = []
        delivered = False
        headers = [(b"host", b"testserver"), (b"content-length", str(len(body)).encode())]
        if content_type:
            headers.append((b"content-type", content_type.encode()))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, object]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        await app(scope, receive, send)
        start = next(message for message in messages if message["type"] == "http.response.start")
        response_body = b"".join(
            message.get("body", b"") for message in messages if message["type"] == "http.response.body"
        )
        response_headers = {
            key.decode("latin-1"): value.decode("latin-1") for key, value in start["headers"]
        }
        return ASGIResponse(status_code=start["status"], headers=response_headers, content=response_body)

    return asyncio.run(invoke())


def test_api_discovery_and_mutating_benchmark_are_not_public() -> None:
    assert request("GET", "/docs").status_code == 404
    assert request("GET", "/redoc").status_code == 404
    assert request("GET", "/openapi.json").status_code == 404
    assert request("POST", "/evaluation/offline/run").status_code == 404
    assert request("GET", "/trace/latest").status_code == 404


def test_live_telemetry_does_not_return_prompt_text() -> None:
    secret_marker = "private-marker-that-must-not-appear"
    query(QueryRequest(query=secret_marker, top_k=1))
    payload = live_monitoring().model_dump()

    assert secret_marker not in json.dumps(payload)
    assert all("query" not in record for record in payload["recent_records"])


def test_large_request_body_is_rejected_before_validation() -> None:
    response = request(
        "POST",
        "/query",
        b"x" * (MAX_REQUEST_BODY_BYTES + 1),
        "application/json",
    )

    assert response.status_code == 413


def test_security_headers_are_added_to_responses() -> None:
    response = request("GET", "/docs")

    assert response.status_code == 404
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_bodyless_routes_preserve_disconnect_events() -> None:
    async def exercise() -> list[str]:
        received_types: list[str] = []
        incoming = iter(
            (
                {"type": "http.request", "body": b"", "more_body": False},
                {"type": "http.disconnect"},
            )
        )

        async def receive() -> dict[str, object]:
            return next(incoming)

        async def send(message: dict[str, object]) -> None:
            return None

        async def downstream(scope, receive, send) -> None:
            received_types.append((await receive())["type"])
            received_types.append((await receive())["type"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = PublicDemoSecurityMiddleware(downstream)
        await middleware(
            {"type": "http", "method": "GET", "path": "/", "headers": []},
            receive,
            send,
        )
        return received_types

    assert asyncio.run(exercise()) == ["http.request", "http.disconnect"]


def test_navigation_does_not_link_to_disabled_api_docs() -> None:
    frontend_dir = REPO_ROOT / "frontend"

    for filename in ("index.html", "app.html", "ops.html", "case-studies.html"):
        assert 'href="/docs"' not in (frontend_dir / filename).read_text(encoding="utf-8")


def test_console_polling_stays_within_shared_telemetry_budget() -> None:
    source = (REPO_ROOT / "frontend" / "ops.js").read_text(encoding="utf-8")
    refresh_ms = int(re.search(r"TELEMETRY_REFRESH_MS = (\d+)", source).group(1))
    telemetry_requests_per_refresh = 3
    concurrent_consoles = 4

    requests_per_minute = (
        concurrent_consoles * telemetry_requests_per_refresh * 60_000 // refresh_ms
    )
    assert requests_per_minute <= TELEMETRY_REQUEST_LIMIT


def test_rate_limiter_rejects_requests_over_its_window() -> None:
    async def exercise() -> list[bool]:
        limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
        return [await limiter.allow(), await limiter.allow(), await limiter.allow()]

    assert asyncio.run(exercise()) == [True, True, False]
