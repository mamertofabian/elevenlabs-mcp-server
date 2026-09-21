from __future__ import annotations

import json
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

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
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


@pytest.mark.parametrize(
    "tool_name, completed",
    [
        ("generate_audio_simple", 0),
        ("generate_audio_script", 0),
        ("generate_audio_script", 1),
    ],
)
def test_legacy_job_persists_and_returns_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str, completed: int
) -> None:
    server = ElevenLabsServer(
        Settings(
            launch_cwd=tmp_path,
            output_dir=tmp_path / "output",
            database_path=tmp_path / "state" / "history.db",
            database_path_explicit=True,
        ),
        environ={"ELEVENLABS_API_KEY": "fixture"},
        enable_revival=False,
    )
    server.api.api_key = None
    calls = []

    class RetainedAudio:
        def export(self, path: Path, format: str) -> None:
            path.write_bytes(b"retained-audio")

    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.AudioSegment.from_mp3", lambda _: RetainedAudio()
    )
    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.time.sleep", lambda _: None)

    def post(*args: object, **kwargs: object) -> _Response:
        calls.append(kwargs)
        if len(calls) <= completed:
            return _Response(200)
        raise ReadTimeout("private timeout detail")

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)

    async def scenario() -> None:
        result = None
        await server.initialize()
        server.api.api_key = "fixture"
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
                    tool_name,
                    {"text": "Fixture"}
                    if tool_name == "generate_audio_simple"
                    else {
                        "script": json.dumps(
                            [{"text": "one"}, {"text": "two"}, {"text": "three"}]
                        )
                    },
                )
            tasks.cancel_scope.cancel()

        assert result is not None
        text = result.content[0]
        assert isinstance(text, TextContent)
        assert "upstream outcome is unknown" in text.text.lower()
        jobs = await server.db.get_all_jobs()
        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        assert jobs[0].error is not None
        assert "upstream outcome is unknown" in jobs[0].error.lower()
        assert "retryable: false" in jobs[0].error.lower()
        assert len(calls) == completed + 1
        assert jobs[0].completed_parts == completed
        if completed:
            assert jobs[0].output_file is not None
            assert Path(jobs[0].output_file).read_bytes() == b"retained-audio"
        else:
            assert jobs[0].output_file is None
        assert "private timeout detail" not in text.text

    anyio.run(scenario)


def test_success_without_request_id_does_not_repeat_synthesis(monkeypatch):
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _Response(200)

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    monkeypatch.setattr(_retry_state(), "sleep", lambda _: None)
    assert api.generate_audio_segment("hello", "voice") == (b"audio", None)
    assert len(calls) == 1


def test_local_write_failure_does_not_repeat_successful_synthesis(
    tmp_path, monkeypatch
):
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _Response(200, {"request-id": "request"})

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    monkeypatch.setattr(_retry_state(), "sleep", lambda _: None)
    with pytest.raises(OSError):
        api.generate_audio_segment("hello", "voice", output_file=str(tmp_path))
    assert len(calls) == 1


@pytest.mark.parametrize("status", [408, 409, 501, 507])
def test_unclassified_http_failure_is_not_replayed(status, monkeypatch):
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _Response(status)

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    monkeypatch.setattr(_retry_state(), "sleep", lambda _: None)
    with pytest.raises(RuntimeError):
        api.generate_audio_segment("hello", "voice")
    assert len(calls) == 1
