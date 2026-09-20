from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tenacity import RetryError

from elevenlabs_mcp.elevenlabs_api import ElevenLabsAPI, VoiceData


def _write_socket_guard(root: Path) -> Path:
    marker = root / "network-attempted"
    (root / "sitecustomize.py").write_text(
        """import os
import socket
from pathlib import Path
marker = Path(os.environ["NETWORK_MARKER"])
def blocked(*args, **kwargs):
    marker.write_text("attempted", encoding="utf-8")
    raise RuntimeError("External network access is disabled")
socket.create_connection = blocked
socket.getaddrinfo = blocked
def guard_socket_method(name):
    original = getattr(socket.socket, name)
    def guarded(self, *args, **kwargs):
        if self.family == socket.AF_UNIX:
            return original(self, *args, **kwargs)
        return blocked(*args, **kwargs)
    return guarded
for method_name in ("connect", "connect_ex", "send", "sendall", "sendto", "sendmsg"):
    setattr(socket.socket, method_name, guard_socket_method(method_name))
""",
        encoding="utf-8",
    )
    return marker


async def _list_tools_without_credentials(tmp_path: Path) -> list[str]:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    work_dir = sandbox / "work"
    home_dir = sandbox / "home"
    temp_dir = sandbox / "tmp"
    for path in (work_dir, home_dir, temp_dir):
        path.mkdir()
    marker = _write_socket_guard(sandbox)
    env = {
        **os.environ,
        "ELEVENLABS_API_KEY": "",
        "ELEVENLABS_OUTPUT_DIR": str(work_dir / "output"),
        "ELEVENLABS_DATABASE_PATH": str(work_dir / "output" / "history.db"),
        "ELEVENLABS_LOG_LEVEL": "ERROR",
        "HOME": str(home_dir),
        "TMPDIR": str(temp_dir),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "PATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(sandbox), os.environ.get("PYTHONPATH")])
        ),
        "NETWORK_MARKER": str(marker),
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "elevenlabs_mcp.server"],
        env=env,
    )
    async with (
        stdio_client(parameters) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
    assert not marker.exists()
    return [tool.name for tool in tools.tools]


async def _list_tools_with_timeout(tmp_path: Path) -> list[str]:
    return await asyncio.wait_for(_list_tools_without_credentials(tmp_path), timeout=10)


def test_real_stdio_discovery_lists_six_tools_without_key_or_ffmpeg(
    tmp_path: Path,
) -> None:
    tool_names = asyncio.run(_list_tools_with_timeout(tmp_path))

    assert tool_names == [
        "generate_audio_simple",
        "generate_audio_script",
        "delete_job",
        "get_audio_file",
        "list_voices",
        "get_voiceover_history",
    ]


def test_api_construction_without_key_preserves_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    for name in (
        "ELEVENLABS_VOICE_ID",
        "ELEVENLABS_MODEL_ID",
        "ELEVENLABS_STABILITY",
        "ELEVENLABS_SIMILARITY_BOOST",
        "ELEVENLABS_STYLE",
    ):
        monkeypatch.delenv(name, raising=False)

    api = ElevenLabsAPI()

    assert api.api_key is None
    assert api.voice_id == "iEw1wkYocsNy7I7pteSN"
    assert api.model_id == "eleven_multilingual_v2"
    assert api.stability == 0.5
    assert api.similarity_boost == 0.75
    assert api.style == 0.1
    assert api.base_url == "https://api.elevenlabs.io/v1"
    assert set(api.MODELS) == {
        "eleven_multilingual_v2",
        "eleven_flash_v2_5",
        "eleven_flash_v2",
    }


def test_missing_key_rejects_metadata_and_synthesis_before_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")

    def unexpected_http(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("HTTP dispatch occurred without an API key")

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.get", unexpected_http)
    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", unexpected_http)
    api = ElevenLabsAPI()
    output_dir = tmp_path / "audio"

    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        api.get_voices()
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        api.generate_audio_segment("Fixture", "voice-a", debug_info=[])
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        api.generate_full_audio([{"text": "Fixture"}], output_dir)
    assert not output_dir.exists()


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.headers = headers or {}
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_fake_http_preserves_voice_and_segment_success_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-key")
    api = ElevenLabsAPI()
    voice_payload = {
        "voices": [
            {
                "voice_id": "voice-a",
                "name": "Fixture Voice",
                "category": "generated",
                "labels": {"accent": "neutral"},
                "description": "Synthetic fixture",
                "preview_url": "https://example.invalid/voice.mp3",
                "high_quality_base_model_ids": ["fixture-model"],
            }
        ]
    }
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(status_code=200, payload=voice_payload),
    )
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=200,
            content=b"synthetic-audio",
            headers={"request-id": "request-a"},
        ),
    )

    voices = api.get_voices()
    voice: VoiceData = voices[0]
    audio, request_id = api.generate_audio_segment(
        "Fixture", voice["voice_id"], debug_info=[]
    )
    category = voice["category"]
    labels = voice["labels"]
    description = voice["description"]
    preview_url = voice["preview_url"]
    high_quality_base_model_ids = voice["high_quality_base_model_ids"]

    assert VoiceData.__required_keys__ == {
        "voice_id",
        "name",
        "category",
        "labels",
        "description",
        "preview_url",
        "high_quality_base_model_ids",
    }
    assert voice["voice_id"] == "voice-a"
    assert voice["name"] == "Fixture Voice"
    assert category == "generated"
    assert labels == {"accent": "neutral"}
    assert description == "Synthetic fixture"
    assert preview_url == "https://example.invalid/voice.mp3"
    assert high_quality_base_model_ids == ["fixture-model"]
    assert audio == b"synthetic-audio"
    assert request_id == "request-a"

    attempts = 0

    class MalformedResponse(_FakeResponse):
        def json(self) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            raise json.JSONDecodeError("malformed", "{", 0)

    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.requests.get",
        lambda *args, **kwargs: MalformedResponse(status_code=200),
    )
    retry_state = cast(Any, ElevenLabsAPI.get_voices).retry
    monkeypatch.setattr(retry_state, "sleep", lambda _: None)

    with pytest.raises(RetryError):
        api.get_voices()

    assert attempts == 3
