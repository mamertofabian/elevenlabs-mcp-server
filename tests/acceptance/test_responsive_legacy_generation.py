from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from anyio import to_thread
from mcp import ClientSession

from elevenlabs_mcp.config import Settings
from elevenlabs_mcp.server import ElevenLabsServer


def _server(tmp_path: Path) -> ElevenLabsServer:
    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "state" / "history.db",
        database_path_explicit=True,
    )
    server = ElevenLabsServer(
        settings, environ={"ELEVENLABS_API_KEY": "fixture"}, enable_revival=False
    )
    server.api.get_voices = list
    return server


async def _run_server_session(server: ElevenLabsServer, scenario: Any) -> None:
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


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("generate_audio_simple", {"text": "Simple fixture"}),
        ("generate_audio_script", {"script": "Script fixture"}),
    ],
)
def test_discovery_and_history_remain_responsive_during_blocked_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    server = _server(tmp_path)
    render_started = threading.Event()

    def blocked_render(
        script_parts: list[dict[str, object]],
        output_dir: Path,
        output_id: str | None = None,
    ) -> tuple[str, list[str], int]:
        assert output_id is not None
        render_started.set()
        time.sleep(0.6)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"full_audio_{output_id}.mp3"
        output_file.write_bytes(b"fixture")
        return str(output_file), [], len(script_parts)

    monkeypatch.setattr(server.api, "generate_full_audio", blocked_render)

    async def scenario(session: ClientSession) -> None:
        results: list[object] = []

        async def generate() -> None:
            results.append(await session.call_tool(tool_name, arguments))

        started = time.perf_counter()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(generate)
            observed = await to_thread.run_sync(render_started.wait, 1)
            assert observed
            tools = await session.list_tools()
            history = await session.read_resource("voiceover://history")
            elapsed = time.perf_counter() - started
            assert len(tools.tools) == 6
            assert history.contents
            assert elapsed < 0.3
            with anyio.fail_after(2):
                while not results:
                    await anyio.sleep(0.01)
            tasks.cancel_scope.cancel()

    anyio.run(_run_server_session, server, scenario)


def test_concurrent_legacy_renders_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path)
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def tracked_render(
        script_parts: list[dict[str, object]],
        output_dir: Path,
        output_id: str | None = None,
    ) -> tuple[str, list[str], int]:
        nonlocal active, maximum_active
        assert output_id is not None
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.15)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"full_audio_{output_id}.mp3"
        output_file.write_bytes(b"fixture")
        with state_lock:
            active -= 1
        return str(output_file), [], len(script_parts)

    monkeypatch.setattr(server.api, "generate_full_audio", tracked_render)

    async def scenario(session: ClientSession) -> None:
        completed = 0

        async def generate(text: str) -> None:
            nonlocal completed
            await session.call_tool("generate_audio_simple", {"text": text})
            completed += 1

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(generate, "first")
            tasks.start_soon(generate, "second")
        assert completed == 2

    anyio.run(_run_server_session, server, scenario)
    assert maximum_active == 1


def test_notification_failure_does_not_stop_following_requests(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)

    @server.server.progress_notification()
    async def fail_notification(
        progress_token: str | int, progress: float, total: float | None
    ) -> None:
        del progress_token, progress, total
        raise RuntimeError("synthetic notification failure")

    async def scenario(session: ClientSession) -> None:
        await session.send_progress_notification("fixture", 0.5, 1.0)
        await anyio.sleep(0.05)
        tools = await session.list_tools()
        assert len(tools.tools) == 6

    anyio.run(_run_server_session, server, scenario)
