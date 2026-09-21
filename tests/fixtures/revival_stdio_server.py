"""Offline process-test entry point. Never selected by the production CLI."""

import json
import os
import time
from pathlib import Path

import httpx

from elevenlabs_mcp.config import Settings
from elevenlabs_mcp.jobs import VoiceoverService
from elevenlabs_mcp.provider import ElevenLabsProvider
from elevenlabs_mcp.server import ElevenLabsServer

root = Path(os.environ["REVIVAL_TEST_ROOT"])
audio = (root / "tone.mp3").read_bytes()
count = 0


def handle(request):
    global count
    if request.method == "POST":
        count += 1
        with (root / "dispatches.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "text": json.loads(request.content).get("text"),
                        "pid": os.getpid(),
                    }
                )
                + "\n"
            )
        if os.environ.get("REVIVAL_CRASH_ON_FOURTH") == "1" and count == 4:
            (root / "inflight.pid").write_text(str(os.getpid()))
            time.sleep(30)
        return httpx.Response(200, content=audio, headers={"request-id": str(count)})
    return httpx.Response(200, json={"voices": [], "has_more": False})


provider = ElevenLabsProvider(
    "offline-fixture", httpx.Client(transport=httpx.MockTransport(handle))
)
settings = Settings(
    launch_cwd=root,
    output_dir=root / "output",
    database_path=root / "state.db",
    database_path_explicit=True,
)
service = VoiceoverService(settings.database_path, settings.output_dir, provider)
server = ElevenLabsServer(
    settings, environ={"ELEVENLABS_API_KEY": ""}, revival_service=service
)
import asyncio

asyncio.run(server.run())
