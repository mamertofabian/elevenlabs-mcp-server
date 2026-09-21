from __future__ import annotations

import asyncio
import io
import itertools
import subprocess
import wave
from array import array

import pytest

from elevenlabs_mcp.jobs import VoiceoverService
from tests.integration.test_revival_end_to_end import Provider, settled


@pytest.mark.asyncio
async def test_final_wave_preserves_tone_order_and_explicit_silence(tmp_path):
    bank = {}
    for frequency in [330, 660, 990]:
        bank[str(frequency)] = (
            await asyncio.to_thread(
                subprocess.run,
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency={frequency}:duration=0.3",
                    "-f",
                    "mp3",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
        ).stdout

    class Tones(Provider):
        def generate(self, request, options):
            self.calls.append(request.chunk_id)
            yield bank[request.chunk.fragments[0].text]

    script = {
        "script_version": "1",
        "cast": {"a": {"voice_id": "voice"}},
        "scenes": [
            {
                "id": "s",
                "parts": [
                    {"id": f"p{i}", "actor": "a", "text": str(f), "pause_after_ms": 200}
                    for i, f in enumerate([330, 660, 990])
                ],
            }
        ],
    }
    options = {
        "engine": "tts",
        "model_id": "eleven_multilingual_v2",
        "export_format": "wav",
    }
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", Tones(b""))
    await service.start()
    try:
        plan = service.plan(script, options)
        job = await service.submit(
            script,
            options,
            plan["plan_hash"],
            "submit",
            {"max_total_characters": 100, "max_total_requests": 3},
        )
        done = await settled(service, job["job_id"])
        assert done["status"] == "completed"
        final = await service.get_artifact(done["final_artifact_ids"][0], "inline")
        with wave.open(io.BytesIO(final["data"]), "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getframerate() == 44100
            samples = array("h", wav.readframes(wav.getnframes()))
        offset = 0
        for chunk, frequency in zip(
            await service.store.chunks(job["job_id"]), [330, 660, 990], strict=True
        ):
            window = samples[offset + 4410 : offset + 8820]
            rising = sum(a <= 0 < b for a, b in itertools.pairwise(window))
            assert abs(rising * 10 - frequency) <= 20
            end = offset + round(chunk["duration_ms"] * 44.1)
            assert max(abs(x) for x in samples[end + 2205 : end + 6615]) == 0
            offset = end + 8820
    finally:
        await service.close()
