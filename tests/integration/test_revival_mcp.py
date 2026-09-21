from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import BlobResourceContents, TextContent

from tests.integration.test_revival_end_to_end import OPTIONS, SCRIPT

RUNNER = Path(__file__).parents[1] / "fixtures/revival_stdio_server.py"


def parameters(root, crash=False):
    return StdioServerParameters(
        command=sys.executable,
        args=[str(RUNNER)],
        env={
            **os.environ,
            "REVIVAL_TEST_ROOT": str(root),
            "REVIVAL_CRASH_ON_FOURTH": "1" if crash else "0",
        },
    )


async def call(session, name, args):
    result = await session.call_tool(name, args)
    envelope = json.loads(result.content[0].text)
    assert not result.is_error, envelope
    from jsonschema import Draft202012Validator, FormatChecker

    definitions = json.loads(
        (
            Path(__file__).parents[2] / "docs/revival/schemas/contracts.schema.json"
        ).read_text()
    )["$defs"]
    model = (
        "PlanView"
        if name == "plan_voiceover"
        else "ArtifactView"
        if name == "get_artifact"
        else "JobListView"
        if name == "list_jobs"
        else "JobView"
    )
    Draft202012Validator(
        {"$ref": "#/$defs/" + model, "$defs": definitions},
        format_checker=FormatChecker(),
    ).validate(envelope["data"])
    return envelope["data"], result


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["tts", "dialogue"])
async def test_real_stdio_process_kill_restart_resume_and_retrieve(tmp_path, engine):
    options = {
        **OPTIONS,
        "engine": engine,
        "model_id": "eleven_v3" if engine == "dialogue" else "eleven_multilingual_v2",
    }
    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-f",
            "mp3",
            str(tmp_path / "tone.mp3"),
        ],
        check=True,
        timeout=10,
    )
    job_id = None
    args = {}
    done = {}
    try:
        async with (
            stdio_client(parameters(tmp_path, True)) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            resources = await session.list_resources()
            assert {str(resource.uri) for resource in resources.resources} == {
                "voiceover://history",
                "voiceover://voices",
            }
            names = {t.name for t in (await session.list_tools()).tools}
            assert {
                "plan_voiceover",
                "submit_voiceover",
                "resume_voiceover",
                "get_artifact",
                "generate_audio_script",
            } <= names
            plan, _ = await call(
                session, "plan_voiceover", {"script": SCRIPT, "options": options}
            )
            args = {
                "script": SCRIPT,
                "options": options,
                "plan_hash": plan["plan_hash"],
                "idempotency_key": "submit",
                "budget": {"max_total_characters": 1000, "max_total_requests": 10},
            }
            job, _ = await call(session, "submit_voiceover", args)
            job_id = job["job_id"]
            for _ in range(400):
                if (tmp_path / "inflight.pid").exists():
                    break
                await asyncio.sleep(0.025)
            assert (tmp_path / "inflight.pid").exists()
            job, _ = await call(session, "get_job", {"job_id": job_id})
            assert job["completed_chunks"] == 3
            os.kill(int((tmp_path / "inflight.pid").read_text()), signal.SIGKILL)
    except BaseExceptionGroup as group:
        # The killed stdio connection may close with an SDK task-group error.
        if job_id is None:
            raise
        assert "AssertionError" not in repr(group)
    assert job_id
    before = (tmp_path / "dispatches.jsonl").read_text().splitlines()
    assert len(before) == 4
    async with (
        stdio_client(parameters(tmp_path)) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        job, _ = await call(session, "get_job", {"job_id": job_id})
        assert job["status"] == "paused" and job["completed_chunks"] == 3
        assert (tmp_path / "dispatches.jsonl").read_text().splitlines() == before
        rejected = await session.call_tool(
            "resume_voiceover",
            {
                "job_id": job_id,
                "expected_revision": 0,
                "idempotency_key": "resume",
                "budget": args["budget"],
            },
        )
        assert isinstance(rejected.content[0], TextContent)
        assert (
            rejected.is_error
            and json.loads(rejected.content[0].text)["error"]["code"]
            == "UPSTREAM_OUTCOME_UNKNOWN"
        )
        await call(
            session,
            "resume_voiceover",
            {
                "job_id": job_id,
                "expected_revision": 0,
                "idempotency_key": "resume",
                "budget": args["budget"],
                "retry_uncertain": True,
            },
        )
        for _ in range(400):
            done, _ = await call(session, "get_job", {"job_id": job_id})
            if done["status"] == "completed":
                break
            assert done["status"] not in {"failed", "paused"}, done
            await asyncio.sleep(0.025)
        assert done["status"] == "completed"
        assert done["completed_parts"] == 5
        assert len((tmp_path / "dispatches.jsonl").read_text().splitlines()) == 6
        artifact, result = await call(
            session,
            "get_artifact",
            {
                "artifact_id": next(
                    a["artifact_id"] for a in done["artifacts"] if a["kind"] == "final"
                ),
                "mode": "inline",
            },
        )
        assert (
            artifact["mime_type"] == "audio/wav"
            and result.content[1].type == "resource"
        )
        resource = await session.read_resource(artifact["uri"])
        assert isinstance(resource.contents[0], BlobResourceContents)
        assert resource.contents[0].blob == result.content[1].resource.blob
        _production, result = await call(
            session,
            "get_artifact",
            {
                "artifact_id": next(
                    a["artifact_id"]
                    for a in done["artifacts"]
                    if a["kind"] == "production_manifest"
                ),
                "mode": "inline",
            },
        )
        import base64

        record = json.loads(base64.b64decode(result.content[1].resource.blob))
        assert len(record["chunks"]) == 5
        assert all(
            chunk["provider_request_id"] is not None for chunk in record["chunks"]
        )
