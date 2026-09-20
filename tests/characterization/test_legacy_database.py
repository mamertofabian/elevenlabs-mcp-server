from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from elevenlabs_mcp.database import Database
from elevenlabs_mcp.models import AudioJob


def test_legacy_job_crud_round_trip(tmp_path: Path) -> None:
    database = Database(tmp_path / "state" / "history.db")
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    job = AudioJob(
        id="fixture-job",
        status="pending",
        script_parts=[{"text": "Fixture narration", "voice_id": None}],
        created_at=created_at,
        updated_at=created_at,
        total_parts=1,
    )

    async def scenario() -> None:
        await database.initialize()
        await database.insert_job(job)
        stored = await database.get_job(job.id)
        assert stored is not None
        assert stored.to_dict() == job.to_dict()

        job.status = "completed"
        job.output_file = "output/fixture.mp3"
        job.completed_parts = 1
        await database.update_job(job)

        jobs = await database.get_all_jobs()
        assert [stored_job.id for stored_job in jobs] == [job.id]
        assert jobs[0].status == "completed"
        assert jobs[0].completed_parts == 1
        assert await database.delete_job(job.id) is True
        assert await database.delete_job(job.id) is False
        assert await database.get_job(job.id) is None
        await database.cleanup()

    asyncio.run(scenario())

    assert str(database.db_path) == str(tmp_path / "state" / "history.db")
    assert not Path(database.db_path).exists()


def test_legacy_voice_cache_round_trip(tmp_path: Path) -> None:
    database = Database(tmp_path / "voices.db")
    voices = [
        {
            "voice_id": "voice-a",
            "name": "Fixture Voice",
            "category": "generated",
            "labels": {"accent": "neutral"},
            "description": "Synthetic fixture",
            "preview_url": "https://example.invalid/fixture.mp3",
            "high_quality_base_model_ids": ["fixture-model"],
        }
    ]

    async def scenario() -> None:
        await database.initialize()
        empty, empty_needs_refresh = await database.get_voices()
        assert empty == []
        assert empty_needs_refresh is True

        await database.upsert_voices(voices)
        cached, fresh_needs_refresh = await database.get_voices(
            Database.CACHE_DURATION_SECONDS
        )
        assert cached == voices
        assert fresh_needs_refresh is False

        stale, stale_needs_refresh = await database.get_voices(
            int(timedelta(microseconds=1).total_seconds())
        )
        assert stale == voices
        assert stale_needs_refresh is False

    asyncio.run(scenario())
