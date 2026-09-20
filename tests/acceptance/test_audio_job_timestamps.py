from __future__ import annotations

from datetime import UTC, datetime

from elevenlabs_mcp.models import AudioJob, ScriptPart


def _job(job_id: str) -> AudioJob:
    return AudioJob(id=job_id, status="pending", script_parts=[{"text": "Fixture"}])


def test_direct_jobs_receive_distinct_aware_utc_defaults() -> None:
    first = _job("first")
    second = _job("second")

    assert first.created_at is not second.created_at
    assert first.updated_at is not second.updated_at
    assert first.created_at.tzinfo is UTC
    assert first.updated_at.tzinfo is UTC
    assert second.created_at.tzinfo is UTC
    assert second.updated_at.tzinfo is UTC
    from elevenlabs_mcp.models import utc_now

    current = utc_now()
    assert current.tzinfo is UTC


def test_create_uses_one_injected_clock_instant() -> None:
    instant = datetime(2026, 9, 20, 1, 2, 3, tzinfo=UTC)
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        return instant

    job = AudioJob.create(
        id="injected",
        status="pending",
        script_parts=[{"text": "Fixture"}],
        clock=clock,
    )

    assert calls == 1
    assert job.created_at == instant
    assert job.updated_at == instant


def test_legacy_naive_timestamps_are_interpreted_as_utc() -> None:
    source = {
        "id": "legacy",
        "status": "completed",
        "script_parts": [{"text": "Historical"}],
        "created_at": "2024-01-02T03:04:05",
        "updated_at": "2024-01-02T04:05:06",
    }
    source_before = dict(source)

    job = AudioJob.from_dict(source)

    assert job.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert job.updated_at == datetime(2024, 1, 2, 4, 5, 6, tzinfo=UTC)
    assert source == source_before


def test_legacy_serialization_field_set_remains_unchanged() -> None:
    part = ScriptPart(text="Fixture", voice_id="voice-a", actor="Narrator")
    job = AudioJob.create(
        id="serialized",
        status="pending",
        script_parts=[
            {"text": part.text, "voice_id": part.voice_id, "actor": part.actor}
        ],
        clock=lambda: datetime(2026, 9, 20, tzinfo=UTC),
    )

    serialized = job.to_dict()
    output_file = job.output_file
    error = job.error

    assert set(serialized) == {
        "id",
        "status",
        "script_parts",
        "output_file",
        "error",
        "created_at",
        "updated_at",
        "total_parts",
        "completed_parts",
    }
    assert serialized["created_at"] == "2026-09-20T00:00:00+00:00"
    assert serialized["updated_at"] == "2026-09-20T00:00:00+00:00"
    assert output_file is None
    assert error is None
