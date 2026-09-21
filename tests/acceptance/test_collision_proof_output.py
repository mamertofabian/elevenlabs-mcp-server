from __future__ import annotations

from pathlib import Path
from uuid import UUID

import anyio
import pytest
from mcp import ClientSession

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
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture-key"})
    payloads = iter((b"first", b"second", b"third", b"fourth"))
    monkeypatch.setattr(
        api,
        "generate_audio_segment",
        lambda **kwargs: (next(payloads), "request-id"),
    )
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.AudioSegment.from_mp3",
        lambda stream: _FakeAudio(stream.read()),
    )
    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.time.sleep", lambda _: None)
    return api


def _render(api: ElevenLabsAPI, output_dir: Path, output_id: str | None = None) -> Path:
    output_file, _, completed = api.generate_full_audio(
        [{"text": "Fixture"}], output_dir, output_id=output_id
    )
    assert completed == 1
    return Path(output_file)


def test_distinct_job_ids_create_distinct_files_in_same_clock_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(monkeypatch)
    first_id = "00000000-0000-4000-8000-000000000001"
    second_id = "00000000-0000-4000-8000-000000000002"

    first = _render(api, tmp_path, first_id)
    second = _render(api, tmp_path, second_id)

    assert first.name == f"full_audio_{first_id}.mp3"
    assert second.name == f"full_audio_{second_id}.mp3"
    assert first != second
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_repeated_job_id_never_overwrites_existing_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(monkeypatch)
    output_id = "00000000-0000-4000-8000-000000000003"
    existing = _render(api, tmp_path, output_id)
    original = existing.read_bytes()

    with pytest.raises(FileExistsError):
        _render(api, tmp_path, output_id)

    assert existing.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_omitted_output_id_uses_unique_uuid_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(monkeypatch)

    first = _render(api, tmp_path)
    second = _render(api, tmp_path)

    first_uuid = UUID(first.stem.removeprefix("full_audio_"))
    second_uuid = UUID(second.stem.removeprefix("full_audio_"))
    assert first_uuid != second_uuid
    assert first.exists()
    assert second.exists()


def test_invalid_output_id_cannot_create_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(monkeypatch)
    output_dir = tmp_path / "not-created"

    for output_id in ("../escape", "not-a-uuid", ""):
        with pytest.raises(ValueError):
            _render(api, output_dir, output_id)

    assert not output_dir.exists()


def test_legacy_server_tools_publish_with_persisted_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "state" / "history.db",
        database_path_explicit=True,
    )
    server = ElevenLabsServer(
        settings, environ={"ELEVENLABS_API_KEY": "fixture"}, enable_revival=False
    )
    captured_ids: list[str] = []

    def fake_generate(
        script_parts: list[dict[str, object]],
        output_dir: Path,
        output_id: str | None = None,
    ) -> tuple[str, list[str], int]:
        assert output_id is not None
        UUID(output_id)
        captured_ids.append(output_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"full_audio_{output_id}.mp3"
        output_file.write_bytes(b"fixture-audio")
        return str(output_file), [], len(script_parts)

    monkeypatch.setattr(server.api, "generate_full_audio", fake_generate)
    monkeypatch.setattr(server.api, "get_voices", list)

    async def scenario() -> None:
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
                await session.call_tool(
                    "generate_audio_simple", {"text": "Simple fixture"}
                )
                await session.call_tool(
                    "generate_audio_script", {"script": "Script fixture"}
                )
            tasks.cancel_scope.cancel()

        jobs = await server.db.get_all_jobs()
        assert {job.id for job in jobs} == set(captured_ids)
        assert all(job.output_file is not None for job in jobs)
        assert {
            Path(job.output_file).stem.removeprefix("full_audio_")
            for job in jobs
            if job.output_file is not None
        } == set(captured_ids)

    anyio.run(scenario)
    assert len(captured_ids) == 2


def test_export_and_link_failures_cleanup_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(monkeypatch)
    output_id = "00000000-0000-4000-8000-000000000004"

    class ExportFailure(_FakeAudio):
        def export(self, path: str | Path, format: str) -> None:
            del path, format
            raise OSError("synthetic export failure")

    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.AudioSegment.from_mp3",
        lambda stream: ExportFailure(stream.read()),
    )
    with pytest.raises(OSError, match="synthetic export failure"):
        _render(api, tmp_path, output_id)
    assert list(tmp_path.iterdir()) == []

    api = _api(monkeypatch)
    preserved = tmp_path / "preserved.txt"
    preserved.write_bytes(b"preserved")
    monkeypatch.setattr(
        "elevenlabs_mcp.elevenlabs_api.os.link",
        lambda source, destination: (_ for _ in ()).throw(
            PermissionError("synthetic link failure")
        ),
    )
    with pytest.raises(PermissionError, match="synthetic link failure"):
        _render(api, tmp_path, output_id)
    assert preserved.read_bytes() == b"preserved"
    assert list(tmp_path.glob("*.tmp")) == []
    assert not (tmp_path / f"full_audio_{output_id}.mp3").exists()
