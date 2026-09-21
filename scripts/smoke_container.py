"""Exercise the built CLI image without credentials, networking, or paid calls."""

from __future__ import annotations

import asyncio
import importlib.metadata
import subprocess
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check_stdio(base: list[str]) -> None:
    expected_version = importlib.metadata.version("elevenlabs-mcp-server")
    for index in range(2):
        parameters = StdioServerParameters(
            command="docker", args=[*base, "-i", "elevenlabs-revival:local"]
        )
        async with (
            stdio_client(parameters) as (read, write),
            ClientSession(read, write) as session,
            asyncio.timeout(30),
        ):
            initialized = await session.initialize()
            assert initialized.server_info.version == expected_version
            assert len((await session.list_tools()).tools) == 15
            script = {
                "script_version": "1",
                "cast": {"a": {"voice_id": "fixture"}},
                "scenes": [
                    {
                        "id": "s",
                        "parts": [
                            {"id": "p", "actor": "a", "text": "Offline planning probe."}
                        ],
                    }
                ],
            }
            options = {"engine": "tts", "model_id": "eleven_multilingual_v2"}
            result = await session.call_tool(
                "plan_voiceover", {"script": script, "options": options}
            )
            plan = result.structured_content["data"]
            assert plan["planned_generation_requests"] == 1
            result = await session.call_tool(
                "submit_voiceover",
                {
                    "script": script,
                    "options": options,
                    "plan_hash": plan["plan_hash"],
                    "idempotency_key": "no-key",
                    "budget": {"max_total_characters": 100, "max_total_requests": 1},
                },
            )
            assert result.is_error
            assert result.structured_content["error"]["code"] == "API_KEY_MISSING"
            jobs = await session.call_tool("list_jobs", {})
            assert jobs.structured_content["data"]["jobs"] == []
        print(f"Container stdio startup {index + 1} passed")


def main() -> None:
    volume = "elevenlabs-revival-smoke-" + uuid4().hex
    subprocess.run(
        ["docker", "volume", "create", volume], check=True, capture_output=True
    )
    base = [
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "-v",
        volume + ":/data",
        "-e",
        "ELEVENLABS_API_KEY=",
    ]
    try:
        asyncio.run(check_stdio(base))
        command = (
            "import os,sqlite3; from pathlib import Path; "
            "from elevenlabs_mcp.assembly import AudioAssembler; "
            "assert os.getuid()!=0; "
            "AudioAssembler(Path('/data/output')).check_dependencies(); "
            "db=sqlite3.connect('/data/voiceover_history.db'); "
            "assert db.execute('PRAGMA user_version').fetchone()[0]==2; "
            "print('Non-root FFmpeg and persisted SQLite passed')"
        )
        result = subprocess.run(
            [
                "docker",
                *base,
                "--entrypoint",
                "/app/.venv/bin/python",
                "elevenlabs-revival:local",
                "-c",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.stdout.strip())
    finally:
        subprocess.run(
            ["docker", "volume", "rm", volume],
            check=True,
            capture_output=True,
            timeout=30,
        )


if __name__ == "__main__":
    main()
