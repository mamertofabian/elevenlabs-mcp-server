from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from mcp import ClientSession
from mcp.types import TextContent
from requests import ConnectionError, ConnectTimeout, ReadTimeout
from tenacity import RetryError

from elevenlabs_mcp.config import Settings
from elevenlabs_mcp.elevenlabs_api import ElevenLabsAPI
from elevenlabs_mcp.server import ElevenLabsServer


class _Response:
    content = b"audio"
    text = ""

    def __init__(
        self, status_code: int, headers: dict[str, str] | None = None
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}


def _retry_state() -> Any:
    return cast(Any, ElevenLabsAPI.generate_audio_segment).retry


def test_numeric_retry_after_controls_bounded_429_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    responses = iter(
        (
            _Response(429, {"Retry-After": "2"}),
            _Response(429, {"Retry-After": "2"}),
            _Response(200, {"request-id": "request-a"}),
        )
    )
    calls = 0
    sleeps: list[float] = []

    def post(*args: object, **kwargs: object) -> _Response:
        nonlocal calls
        del args, kwargs
        calls += 1
        return next(responses)

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    monkeypatch.setattr(_retry_state(), "sleep", sleeps.append)

    result = api.generate_audio_segment("Fixture", "voice-a", debug_info=[])

    assert result == (b"audio", "request-a")
    assert calls == 3
    assert sleeps == [2.0, 2.0]


def test_read_timeout_and_reset_dispatch_once_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elevenlabs_mcp.elevenlabs_api import UpstreamOutcomeUnknownError

    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    sentinel = "PRIVATE_TIMEOUT_SENTINEL"
    monkeypatch.setattr(_retry_state(), "sleep", lambda _: None)

    for exception in (ReadTimeout(sentinel), ConnectionError(sentinel)):
        calls = 0

        def post(
            *args: object, failure: Exception = exception, **kwargs: object
        ) -> _Response:
            nonlocal calls
            del args, kwargs
            calls += 1
            raise failure

        monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
        with pytest.raises(UpstreamOutcomeUnknownError) as captured:
            api.generate_audio_segment("Fixture", "voice-a", debug_info=[])

        error = captured.value
        operation = error.operation
        cause_type = error.cause_type
        retryable = error.retryable
        assert calls == 1
        assert operation == "synthesis"
        assert cause_type == type(exception).__name__
        assert retryable is False
        assert sentinel not in str(error)


def test_connect_timeout_retains_three_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    calls = 0
    monkeypatch.setattr(_retry_state(), "sleep", lambda _: None)

    def post(*args: object, **kwargs: object) -> _Response:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise ConnectTimeout("synthetic connect timeout")

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    with pytest.raises(RetryError):
        api.generate_audio_segment("Fixture", "voice-a", debug_info=[])

    assert calls == 3


def test_legacy_job_persists_and_returns_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from elevenlabs_mcp.elevenlabs_api import UpstreamOutcomeUnknownError

    server = ElevenLabsServer(
        Settings(
            launch_cwd=tmp_path,
            output_dir=tmp_path / "output",
            database_path=tmp_path / "state" / "history.db",
            database_path_explicit=True,
        ),
        environ={"ELEVENLABS_API_KEY": "fixture"},
    )
    server.api.api_key = None
    monkeypatch.setattr(
        server.api,
        "generate_full_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            UpstreamOutcomeUnknownError("synthesis", "ReadTimeout")
        ),
    )

    async def scenario() -> None:
        await server.initialize()
        client_send, server_read = anyio.create_memory_object_stream(10)
        server_send, client_read = anyio.create_memory_object_stream(10)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(
                server.server.run,
                server_read,
                server_send,
                server.server.create_initialization_options(),
            )
            async with ClientSession(client_read, client_send) as session:
                await session.initialize()
                result = await session.call_tool(
                    "generate_audio_simple", {"text": "Fixture"}
                )
            tasks.cancel_scope.cancel()

        text = result.content[0]
        assert isinstance(text, TextContent)
        assert "upstream outcome is unknown" in text.text.lower()
        jobs = await server.db.get_all_jobs()
        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        assert jobs[0].error is not None
        assert "upstream outcome is unknown" in jobs[0].error.lower()
        assert "retryable: false" in jobs[0].error.lower()

    anyio.run(scenario)
