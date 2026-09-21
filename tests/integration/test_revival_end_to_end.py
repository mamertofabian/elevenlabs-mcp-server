from __future__ import annotations

import asyncio

import pytest

SCRIPT = {
    "script_version": "1",
    "cast": {"a": {"voice_id": "voice-a"}, "b": {"voice_id": "voice-b"}},
    "scenes": [
        {
            "id": "scene",
            "parts": [
                {
                    "id": f"p{i}",
                    "actor": "a" if i % 2 == 0 else "b",
                    "text": f"Line {i}.",
                    "pause_after_ms": 100,
                }
                for i in range(5)
            ],
        }
    ],
}
OPTIONS = {
    "engine": "tts",
    "model_id": "eleven_multilingual_v2",
    "export_format": "wav",
}


class Provider:
    def __init__(self, audio, fail_at=None):
        self.context_id = "injected-provider"
        self.audio = audio
        self.calls = []
        self.fail_at = fail_at

    def check_ready(self, options):
        pass

    def generate(self, request, options):
        from elevenlabs_mcp.provider import ProviderError

        self.calls.append(request.chunk_id)
        if len(self.calls) == self.fail_at:
            raise ProviderError("UPSTREAM_OUTCOME_UNKNOWN", uncertain=True)
        yield self.audio

    def close(self):
        pass


async def settled(service, job_id):
    for _ in range(400):
        job = await service.get_job(job_id)
        if job["status"] not in {"queued", "running", "assembling"}:
            return job
        await asyncio.sleep(0.025)
    raise AssertionError("job did not settle")


@pytest.mark.asyncio
async def test_full_revival_restart_reuses_verified_chunks_and_finishes(
    tmp_path, audio
):
    from elevenlabs_mcp.jobs import VoiceoverService

    first = Provider(audio, 4)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "output", first)
    await service.start()
    plan = service.plan(SCRIPT, OPTIONS)
    args = {
        "script": SCRIPT,
        "options": OPTIONS,
        "plan_hash": plan["plan_hash"],
        "idempotency_key": "submit",
        "budget": {"max_total_characters": 1000, "max_total_requests": 10},
    }
    job = await service.submit(**args)
    paused = await settled(service, job["job_id"])
    assert paused["status"] == "paused"
    assert paused["verified_chunks"] == 3
    await service.close()
    second = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "output", second)
    await service.start()
    try:
        assert second.calls == []
        replay = await service.submit(**args)
        assert replay["job_id"] == job["job_id"] and second.calls == []
        with pytest.raises(Exception, match="UPSTREAM_OUTCOME_UNKNOWN"):
            await service.resume(job["job_id"], 0, "resume", args["budget"])
        await service.resume(
            job["job_id"], 0, "resume", args["budget"], retry_uncertain=True
        )
        done = await settled(service, job["job_id"])
        assert done["status"] == "completed", done
        assert done["verified_parts"] == 5
        assert done["reserved_requests"] == 6
        assert len(second.calls) == 2
        assert not set(second.calls) & set(first.calls[:3])
        artifact = await service.get_artifact(done["final_artifact_ids"][0], "inline")
        assert artifact["data"].startswith(b"RIFF")
        assert artifact["duration_ms"] >= 1000
        assert (await service.list_jobs())["jobs"][0]["job_id"] == job["job_id"]
    finally:
        await service.close()
