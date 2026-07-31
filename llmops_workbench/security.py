"""Small, dependency-free safeguards for the public demo service."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


ASGIApp = Callable[[dict[str, Any], Callable[[], Awaitable[dict[str, Any]]], Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[None]]
MAX_REQUEST_BODY_BYTES = 16 * 1024
TELEMETRY_REQUEST_LIMIT = 120

SECURITY_HEADERS = (
    (b"content-security-policy", b"default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; img-src 'self' data:; object-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), geolocation=(), microphone=()"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
)


@dataclass
class SlidingWindowLimiter:
    """Bound aggregate request volume for one process."""

    limit: int
    window_seconds: float
    timestamps: deque[float] = field(default_factory=deque)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def allow(self) -> bool:
        async with self.lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            while self.timestamps and self.timestamps[0] <= cutoff:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.limit:
                return False
            self.timestamps.append(now)
            return True


class PublicDemoSecurityMiddleware:
    """Enforce body limits, aggregate throttling, and browser safeguards."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.query_limiter = SlidingWindowLimiter(limit=30, window_seconds=60.0)
        self.telemetry_limiter = SlidingWindowLimiter(
            limit=TELEMETRY_REQUEST_LIMIT,
            window_seconds=60.0,
        )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        limiter = self.query_limiter if method == "POST" and path == "/query" else None
        if limiter is None and path in {"/metrics", "/observability/summary", "/evaluation/live"}:
            limiter = self.telemetry_limiter
        if limiter is not None and not await limiter.allow():
            await _send_plain_response(send, 429, b"Too many requests", retry_after=True)
            return

        async def secure_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(SECURITY_HEADERS)
            await send(message)

        if method != "POST" or path != "/query":
            await self.app(scope, receive, secure_send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length and _content_length_exceeds_limit(content_length):
            await _send_plain_response(send, 413, b"Request body too large")
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_REQUEST_BODY_BYTES:
                await _send_plain_response(send, 413, b"Request body too large")
                return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, secure_send)


def _content_length_exceeds_limit(raw_value: bytes) -> bool:
    try:
        return int(raw_value) > MAX_REQUEST_BODY_BYTES
    except ValueError:
        return True


async def _send_plain_response(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    status: int,
    body: bytes,
    *,
    retry_after: bool = False,
) -> None:
    headers = [(b"content-type", b"text/plain; charset=utf-8"), *SECURITY_HEADERS]
    if retry_after:
        headers.append((b"retry-after", b"60"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
