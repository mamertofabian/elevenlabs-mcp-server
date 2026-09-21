from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import UUID

import anyio
import pytest
from mcp import ClientSession
from mcp.types import TextContent

from elevenlabs_mcp.config import Settings
from elevenlabs_mcp.elevenlabs_api import ElevenLabsAPI
from elevenlabs_mcp.server import ElevenLabsServer


class _FakeAudio:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __add__(self, other: _FakeAudio) -> _FakeAudio:
        return _FakeAudio(self.payload + other.payload)

    def export(self, path: str | Path, format: str) -> None:
        assert format == "mp3"
        Path(path).write_bytes(self.payload)


def _api(monkeypatch: pytest.MonkeyPatch) -> ElevenLabsAPI:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.AudioSegment.from_mp3",
        lambda stream: _FakeAudio(stream.read()),
    )
    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.time.sleep", lambda _: None)
    return api


def test_middle_part_failure_raises_with_explicit_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(monkeypatch)
    attempts = 0

    def generate(**kwargs: object) -> tuple[bytes, str]:
        nonlocal attempts
        del kwargs
        index = attempts
        attempts += 1
        if index == 1:
            raise RuntimeError("synthetic middle failure")
        return f"part-{index}".encode(), f"request-{index}"

    monkeypatch.setattr(api, "generate_audio_segment", generate)
    output_id = "00000000-0000-4000-8000-000000000010"

    with pytest.raises(RuntimeError) as captured:
        api.generate_full_audio(
            [{"text": "one"}, {"text": "two"}, {"text": "three"}],
            tmp_path,
            output_id=output_id,
        )

    from elevenlabs_mcp.elevenlabs_api import PartialGenerationError

    error = captured.value
    assert isinstance(error, PartialGenerationError)
    assert error.completed_parts == 2
    assert error.failed_part_indexes == (1,)
    assert error.partial_output_file is not None
    partial = Path(error.partial_output_file)
    assert partial.name == f"partial_audio_{output_id}.mp3"
    assert partial.read_bytes() == b"part-0part-2"
    assert list(tmp_path.glob("full_audio_*.mp3")) == []


def test_all_parts_failed_raises_without_publishing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from elevenlabs_mcp.elevenlabs_api import PartialGenerationError

    api = _api(monkeypatch)
    monkeypatch.setattr(
        api,
        "generate_audio_segment",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(PartialGenerationError) as captured:
        api.generate_full_audio(
            [{"text": "one"}, {"text": "two"}],
            tmp_path,
            output_id="00000000-0000-4000-8000-000000000011",
        )

    error = captured.value
    assert error.completed_parts == 0
    assert error.failed_part_indexes == (0, 1)
    assert error.partial_output_file is None
    assert list(tmp_path.iterdir()) == []


def test_legacy_tool_persists_failed_partial_job_without_success_substring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from elevenlabs_mcp.elevenlabs_api import PartialGenerationError

    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "state" / "history.db",
        database_path_explicit=True,
    )
    server = ElevenLabsServer(
        settings, environ={"ELEVENLABS_API_KEY": "fixture"}, enable_revival=False
    )
    server.api.api_key = None
    partial_path = settings.output_dir / "partial_audio_fixture.mp3"

    def fail_with_partial(
        script_parts: list[dict[str, object]],
        output_dir: Path,
        output_id: str | None = None,
    ) -> tuple[str, list[str], int]:
        del script_parts, output_dir
        assert output_id is not None
        UUID(output_id)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_bytes(b"partial")
        raise PartialGenerationError(str(partial_path), 2, (1,))

    monkeypatch.setattr(server.api, "generate_full_audio", fail_with_partial)

    async def scenario() -> None:
        result = None
        await server.initialize()
        client_send, server_read = anyio.create_memory_object_stream(1)
        server_send, client_read = anyio.create_memory_object_stream(1)
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
                    "generate_audio_script",
                    {"script": ('[{"text":"one"},{"text":"two"},{"text":"three"}]')},
                )
            tasks.cancel_scope.cancel()

        assert result is not None
        text = result.content[0]
        assert isinstance(text, TextContent)
        first_line = text.text.splitlines()[0]
        assert first_line == "Error generating audio. Debug info:"
        assert "successful" not in first_line
        assert "unsuccessful" not in first_line
        jobs = await server.db.get_all_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.status == "failed"
        assert job.completed_parts == 2
        assert job.output_file == str(partial_path)
        assert job.error is not None
        assert "failed part indexes: 1" in job.error

    anyio.run(scenario)


@pytest.mark.parametrize("assembly_fails", [False, True])
def test_unknown_outcome_stops_dispatch_and_preserves_completed_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assembly_fails: bool
) -> None:
    from requests import ReadTimeout

    from elevenlabs_mcp.elevenlabs_api import UpstreamOutcomeUnknownError

    api = _api(monkeypatch)
    calls = []
    if assembly_fails:

        def fail_export(*args: object, **kwargs: object) -> None:
            raise OSError("synthetic disk failure")

        monkeypatch.setattr(_FakeAudio, "export", fail_export)

    class Response:
        status_code = 200
        content = b"first-part"
        headers: ClassVar[dict[str, str]] = {}

    def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return Response()
        raise ReadTimeout("private")

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    monkeypatch.setattr(
        cast(Any, ElevenLabsAPI.generate_audio_segment).retry, "sleep", lambda _: None
    )
    with pytest.raises(UpstreamOutcomeUnknownError) as caught:
        api.generate_full_audio(
            [{"text": "one"}, {"text": "two"}, {"text": "three"}], tmp_path
        )
    assert len(calls) == 2
    assert "previous_request_ids" not in calls[1]["json"]
    assert caught.value.completed_parts == 1
    if assembly_fails:
        assert caught.value.partial_output_file is None
        assert isinstance(caught.value.__cause__, OSError)
        assert not list(tmp_path.iterdir())
    else:
        assert caught.value.partial_output_file is not None
        assert Path(caught.value.partial_output_file).read_bytes() == b"first-part"
    assert "retryable: false" in str(caught.value)
    assert not list(tmp_path.glob("full_audio_*.mp3"))
