from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, cast

import anyio
import pytest
from mcp import ClientSession
from mcp.types import TextContent, TextResourceContents
from pydantic import AnyUrl
from requests import ConnectionError
from tenacity import RetryError

from elevenlabs_mcp.config import Settings
from elevenlabs_mcp.elevenlabs_api import ElevenLabsAPI
from elevenlabs_mcp.server import ElevenLabsServer

SCRIPT_SENTINEL = "PRIVATE_SCRIPT_SENTINEL"
VOICE_SENTINEL = "PRIVATE_VOICE_SENTINEL"
ACTOR_SENTINEL = "PRIVATE_ACTOR_SENTINEL"
KEY_SENTINEL = "PRIVATE_KEY_SENTINEL"
BODY_SENTINEL = "PRIVATE_PROVIDER_BODY_SENTINEL"


async def _with_session(server: ElevenLabsServer, scenario: Any) -> None:
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
            await scenario(session)
        tasks.cancel_scope.cancel()


def _server(tmp_path: Path) -> ElevenLabsServer:
    return ElevenLabsServer(
        Settings(
            launch_cwd=tmp_path,
            output_dir=tmp_path / "output",
            database_path=tmp_path / "state" / "history.db",
            database_path_explicit=True,
        ),
        environ={"ELEVENLABS_API_KEY": KEY_SENTINEL},
    )


def test_invalid_log_level_never_writes_protocol_stdout(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "ELEVENLABS_LOG_LEVEL": "INVALID_SENTINEL_LEVEL",
        "ELEVENLABS_API_KEY": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", "import elevenlabs_mcp.server"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert "Invalid log level" in result.stderr


def test_parser_debug_never_echoes_script_or_voice_values(tmp_path: Path) -> None:
    server = _server(tmp_path)
    raw = (
        '[{"text":"PRIVATE_SCRIPT_SENTINEL",'
        '"voice_id":"PRIVATE_VOICE_SENTINEL",'
        '"actor":"PRIVATE_ACTOR_SENTINEL"}]'
    )

    parts, debug = server.parse_script(raw)

    assert parts == [
        {
            "text": SCRIPT_SENTINEL,
            "voice_id": VOICE_SENTINEL,
            "actor": ACTOR_SENTINEL,
        }
    ]
    debug_text = "\n".join(debug)
    assert SCRIPT_SENTINEL not in debug_text
    assert VOICE_SENTINEL not in debug_text
    assert ACTOR_SENTINEL not in debug_text


class _Response:
    status_code = 422
    text = BODY_SENTINEL
    content = b""
    headers: ClassVar[dict[str, str]] = {}


def test_provider_rejection_redacts_body_request_and_credentials(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": KEY_SENTINEL})
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.post",
        lambda *args, **kwargs: _Response(),
    )
    caplog.set_level(logging.DEBUG)

    with pytest.raises(RuntimeError) as captured:
        api.generate_audio_segment(
            SCRIPT_SENTINEL,
            VOICE_SENTINEL,
            debug_info=[],
        )

    combined = str(captured.value) + "\n" + caplog.text
    assert "422" in combined
    assert BODY_SENTINEL not in combined
    assert SCRIPT_SENTINEL not in combined
    assert VOICE_SENTINEL not in combined
    assert KEY_SENTINEL not in combined

    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.get",
        lambda *args, **kwargs: _Response(),
    )
    caplog.clear()
    with pytest.raises(RuntimeError) as metadata_error:
        api.get_voices()
    metadata_combined = str(metadata_error.value) + "\n" + caplog.text
    assert "422" in metadata_combined
    assert BODY_SENTINEL not in metadata_combined
    assert KEY_SENTINEL not in metadata_combined


def test_legacy_success_line_is_preserved_without_untrusted_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path)
    server.api.api_key = None

    def generate(
        script_parts: list[dict[str, object]],
        output_dir: Path,
        output_id: str | None = None,
    ) -> tuple[str, list[str], int]:
        assert output_id is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"full_audio_{output_id}.mp3"
        output_file.write_bytes(b"fixture")
        return str(output_file), [SCRIPT_SENTINEL, KEY_SENTINEL], len(script_parts)

    monkeypatch.setattr(server.api, "generate_full_audio", generate)

    async def scenario(session: ClientSession) -> None:
        result = await session.call_tool(
            "generate_audio_simple", {"text": SCRIPT_SENTINEL}
        )
        text = result.content[0]
        assert isinstance(text, TextContent)
        assert text.text.splitlines()[0] == "Audio generation successful. Debug info:"
        assert SCRIPT_SENTINEL not in text.text
        assert KEY_SENTINEL not in text.text

    anyio.run(_with_session, server, scenario)


def test_general_generation_failure_is_sanitized_in_job_and_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path)
    server.api.api_key = None
    failure = f"{BODY_SENTINEL} {SCRIPT_SENTINEL} {KEY_SENTINEL}"
    monkeypatch.setattr(
        server.api,
        "generate_full_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(failure)),
    )

    async def scenario(session: ClientSession) -> None:
        result = await session.call_tool(
            "generate_audio_simple", {"text": SCRIPT_SENTINEL}
        )
        text = result.content[0]
        assert isinstance(text, TextContent)
        assert text.text.splitlines()[0] == "Error generating audio. Debug info:"
        assert BODY_SENTINEL not in text.text
        assert SCRIPT_SENTINEL not in text.text
        assert KEY_SENTINEL not in text.text
        jobs = await server.db.get_all_jobs()
        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        assert jobs[0].error == "Audio generation failed"

    anyio.run(_with_session, server, scenario)


def test_startup_failure_never_writes_exception_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _server(tmp_path)
    sentinel = "PRIVATE_STARTUP_EXCEPTION_SENTINEL"

    async def fail_cache(*args: object, **kwargs: object) -> tuple[list[dict], bool]:
        del args, kwargs
        raise RuntimeError(sentinel)

    monkeypatch.setattr(server.db, "get_voices", fail_cache)
    caplog.set_level(logging.ERROR)
    anyio.run(server.initialize)
    assert sentinel not in caplog.text
    assert "RuntimeError" in caplog.text
    caplog.clear()

    async def fail_initialize() -> None:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(server, "initialize", fail_initialize)

    with pytest.raises(RuntimeError, match=sentinel):
        anyio.run(server.run)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert sentinel not in captured.err
    assert sentinel not in caplog.text
    assert "RuntimeError" in caplog.text


def test_voice_network_failure_is_sanitized_in_api_and_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "PRIVATE_NETWORK_EXCEPTION_SENTINEL"
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": KEY_SENTINEL})
    retry_state = cast(Any, ElevenLabsAPI.get_voices).retry
    monkeypatch.setattr(retry_state, "sleep", lambda _: None)
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError(sentinel)),
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(RetryError) as captured:
        api.get_voices()

    combined = str(captured.value) + "\n" + caplog.text
    assert sentinel not in combined
    assert "ConnectionError" in combined

    class MalformedResponse(_Response):
        status_code = 200

        def json(self) -> dict[str, Any]:
            raise ValueError(sentinel)

    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.get",
        lambda *args, **kwargs: MalformedResponse(),
    )
    caplog.clear()
    with pytest.raises(RetryError) as malformed:
        api.get_voices()
    malformed_combined = str(malformed.value) + "\n" + caplog.text
    assert sentinel not in malformed_combined
    assert "ValueError" in malformed_combined

    server = _server(tmp_path)
    server.api.api_key = None

    async def fail_cache(*args: object, **kwargs: object) -> tuple[list[dict], bool]:
        del args, kwargs
        raise RuntimeError(sentinel)

    async def scenario(session: ClientSession) -> None:
        monkeypatch.setattr(server.db, "get_voices", fail_cache)
        result = await session.call_tool("list_voices", {})
        text = result.content[0]
        assert isinstance(text, TextContent)
        assert sentinel not in text.text
        assert "error" in text.text
        resource = await session.read_resource(AnyUrl("voiceover://voices"))
        resource_text = resource.contents[0]
        assert isinstance(resource_text, TextResourceContents)
        assert sentinel not in resource_text.text
        assert "error" in resource_text.text

    anyio.run(_with_session, server, scenario)
    assert sentinel not in caplog.text
