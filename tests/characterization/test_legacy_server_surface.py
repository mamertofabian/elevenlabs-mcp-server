from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import mcp.server.stdio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    ClientRequest,
    ListResourceTemplatesRequest,
    ListResourceTemplatesResult,
    TextResourceContents,
)
from pydantic import AnyUrl

from elevenlabs_mcp.database import CREATE_VOICES_TABLE
from elevenlabs_mcp.server import ElevenLabsServer, main


async def _discover_server(
    tmp_path: Path,
) -> tuple[list[str], list[str], str]:
    output_dir = tmp_path / "output"
    database_path = output_dir / "voiceover_history.db"
    output_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(CREATE_VOICES_TABLE)
        await connection.execute(
            """
            INSERT INTO voices (
                voice_id, name, category, labels, description, preview_url,
                high_quality_base_model_ids, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "voice-a",
                "Fixture Voice",
                "generated",
                "{}",
                "Synthetic fixture",
                "",
                "[]",
                "9999-12-31T00:00:00",
            ),
        )
        await connection.commit()

    env = {
        **os.environ,
        "ELEVENLABS_API_KEY": "test-not-a-real-key",
        "ELEVENLABS_OUTPUT_DIR": str(output_dir),
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
        templates = await session.send_request(
            ClientRequest(
                ListResourceTemplatesRequest(method="resources/templates/list")
            ),
            ListResourceTemplatesResult,
        )
        history = await session.read_resource(AnyUrl("voiceover://history"))
        history_content = history.contents[0]
        assert isinstance(history_content, TextResourceContents)
        return (
            [tool.name for tool in tools.tools],
            [str(template.uriTemplate) for template in templates.resourceTemplates],
            history_content.text,
        )


async def _discover_server_with_timeout(
    tmp_path: Path,
) -> tuple[list[str], list[str], str]:
    return await asyncio.wait_for(_discover_server(tmp_path), timeout=5)


def test_legacy_server_registers_six_tools_and_three_resource_uris(
    tmp_path: Path,
) -> None:
    server = ElevenLabsServer()
    assert server.api is not None
    assert server.setup_tools() is None
    assert server.setup_resources() is None
    assert server.setup_notifications() is None

    tool_names, templates, history_text = asyncio.run(
        _discover_server_with_timeout(tmp_path)
    )

    assert tool_names == [
        "generate_audio_simple",
        "generate_audio_script",
        "delete_job",
        "get_audio_file",
        "list_voices",
        "get_voiceover_history",
    ]
    assert templates == ["voiceover://history/{job_id}", "voiceover://voices"]
    assert history_text == "[]"


def test_legacy_server_parses_supported_script_forms(tmp_path: Path) -> None:
    del tmp_path
    server = ElevenLabsServer()

    plain, _ = server.parse_script("Fixture narration")
    direct, _ = server.parse_script('[{"text":"Direct","actor":"Narrator"}]')
    wrapped, _ = server.parse_script(
        '{"script":[{"text":"Wrapped","voice_id":"voice-a"}]}'
    )

    assert plain == [{"text": "Fixture narration", "voice_id": None, "actor": None}]
    assert direct == [{"text": "Direct", "voice_id": None, "actor": "Narrator"}]
    assert wrapped == [{"text": "Wrapped", "voice_id": "voice-a", "actor": None}]


def test_legacy_stdio_entrypoint_discovers_tools_without_provider_network(
    tmp_path: Path, monkeypatch: object
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)

    tool_names, _, _ = asyncio.run(_discover_server_with_timeout(tmp_path))

    assert len(tool_names) == 6
    assert "generate_audio_simple" in tool_names

    run_events: list[str] = []
    server = ElevenLabsServer()

    async def fake_initialize() -> None:
        run_events.append("initialized")

    @asynccontextmanager
    async def fake_stdio_server() -> AsyncIterator[tuple[object, object]]:
        run_events.append("stdio-open")
        yield object(), object()

    async def fake_sdk_run(
        read_stream: object, write_stream: object, options: object
    ) -> None:
        del read_stream, write_stream, options
        run_events.append("sdk-run")

    monkeypatch.setattr(server, "initialize", fake_initialize)
    monkeypatch.setattr(server.server, "run", fake_sdk_run)
    monkeypatch.setattr(mcp.server.stdio, "stdio_server", fake_stdio_server)

    asyncio.run(server.run())

    assert run_events == ["initialized", "stdio-open", "sdk-run"]

    async def fake_entrypoint_run(self: ElevenLabsServer) -> None:
        del self
        run_events.append("main-run")

    monkeypatch.setattr(ElevenLabsServer, "run", fake_entrypoint_run)
    main()

    assert run_events[-1] == "main-run"
