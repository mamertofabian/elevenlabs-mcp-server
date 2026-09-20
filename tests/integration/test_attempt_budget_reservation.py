from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

STARTED = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)


def _plan():
    from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
    from elevenlabs_mcp.planner import ScriptPlanner

    script = Script.model_validate(
        {
            "script_version": "1",
            "cast": {"actor": {"voice_id": "voice"}},
            "scenes": [
                {
                    "id": "scene",
                    "parts": [{"id": "part", "actor": "actor", "text": "hello"}],
                }
            ],
        }
    )
    return ScriptPlanner().plan(
        script,
        VoiceoverOptions(engine="tts", model_id="eleven_multilingual_v2"),
        PlanningLimits(max_text_characters=3),
    )


async def _created_store(path: Path, max_requests: int = 10):
    from elevenlabs_mcp.database import JobStore

    store = JobStore(path)
    await store.initialize()
    plan = _plan()
    await store.create_job("ws", "job", plan, "key", 100, max_requests, STARTED)
    return store, plan


@pytest.mark.asyncio
async def test_reserve_attempt_atomically_updates_job_chunk_and_ledger(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.contracts import AttemptReservation

    store, plan = await _created_store(tmp_path / "db.sqlite")
    chunk = plan.requests[0].chunk
    result = await store.reserve_attempt(
        "job", plan.requests[0].chunk_id, "attempt-1", STARTED
    )
    assert result == AttemptReservation(
        attempt_id="attempt-1",
        job_id="job",
        chunk_id=plan.requests[0].chunk_id,
        reserved_characters=chunk.character_count,
        reserved_requests=1,
        replayed=False,
    )
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT reserved_characters,reserved_requests FROM voiceover_jobs"
        ).fetchone() == (chunk.character_count, 1)
        assert db.execute(
            "SELECT dispatch_state,reserved_characters,started_at,outcome_unknown FROM generation_attempts"
        ).fetchone() == ("reserved", chunk.character_count, STARTED.isoformat(), 0)
        assert db.execute(
            "SELECT status FROM voiceover_chunks WHERE job_id='job'"
        ).fetchone() == ("pending",)


@pytest.mark.asyncio
async def test_attempt_id_replay_is_exact_and_conflicting_owner_fails(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import AttemptConflictError

    store, plan = await _created_store(tmp_path / "db.sqlite")
    first = await store.reserve_attempt(
        "job", plan.requests[0].chunk_id, "attempt", STARTED
    )
    replay = await store.reserve_attempt(
        "job", plan.requests[0].chunk_id, "attempt", STARTED.replace(tzinfo=None)
    )
    assert replay.model_copy(update={"replayed": False}) == first
    assert replay.replayed is True
    with pytest.raises(AttemptConflictError) as caught:
        await store.reserve_attempt(
            "job", plan.requests[1].chunk_id, "attempt", STARTED
        )
    assert caught.value.attempt_id == "attempt"


@pytest.mark.asyncio
async def test_concurrent_reservations_cannot_exceed_cumulative_ceiling(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import BudgetExceededError, JobStore

    path = tmp_path / "db.sqlite"
    store, plan = await _created_store(path, max_requests=2)
    chunk_id = plan.requests[0].chunk_id
    await store.reserve_attempt("job", chunk_id, "initial", STARTED)
    results = await asyncio.gather(
        JobStore(path).reserve_attempt("job", chunk_id, "a1", STARTED),
        JobStore(path).reserve_attempt("job", chunk_id, "a2", STARTED),
        return_exceptions=True,
    )
    assert sum(not isinstance(item, BaseException) for item in results) == 1
    error = next(item for item in results if isinstance(item, BudgetExceededError))
    assert error.max_total_requests == 2 and error.required_requests == 3
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM generation_attempts").fetchone() == (2,)
        assert db.execute(
            "SELECT reserved_requests FROM voiceover_jobs"
        ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_missing_job_or_chunk_and_invalid_attempt_leave_ledger_unchanged(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import ChunkNotFoundError, JobNotFoundError

    store, plan = await _created_store(tmp_path / "db.sqlite")
    with pytest.raises(JobNotFoundError) as missing_job:
        await store.reserve_attempt(
            "missing", plan.requests[0].chunk_id, "attempt", STARTED
        )
    assert missing_job.value.job_id == "missing"
    with pytest.raises(ChunkNotFoundError) as missing_chunk:
        await store.reserve_attempt("job", "missing", "attempt", STARTED)
    assert (missing_chunk.value.job_id, missing_chunk.value.chunk_id) == (
        "job",
        "missing",
    )
    with pytest.raises(ValidationError):
        await store.reserve_attempt("job", plan.requests[0].chunk_id, "", STARTED)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM generation_attempts").fetchone() == (0,)
        assert db.execute(
            "SELECT reserved_characters,reserved_requests FROM voiceover_jobs"
        ).fetchone() == (0, 0)
        assert db.execute(
            "SELECT DISTINCT status FROM voiceover_chunks"
        ).fetchall() == [("pending",)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_status", "chunk_status"),
    [
        ("completed", "pending"),
        ("cancelled", "pending"),
        ("paused", "pending"),
        ("queued", "succeeded"),
        ("running", "unknown"),
    ],
)
async def test_ineligible_job_or_chunk_state_is_preserved_without_reservation(
    tmp_path: Path, job_status: str, chunk_status: str
) -> None:
    from elevenlabs_mcp.database import ReservationStateError

    store, plan = await _created_store(tmp_path / f"{job_status}-{chunk_status}.sqlite")
    chunk_id = plan.requests[0].chunk_id
    async with store.connect() as db:
        await db.execute("UPDATE voiceover_jobs SET status = ?", (job_status,))
        await db.execute(
            "UPDATE voiceover_chunks SET status = ? WHERE job_id = ? AND chunk_id = ?",
            (chunk_status, "job", chunk_id),
        )
        await db.commit()

    with pytest.raises(ReservationStateError) as caught:
        await store.reserve_attempt("job", chunk_id, "attempt", STARTED)

    assert (
        caught.value.job_id,
        caught.value.chunk_id,
        caught.value.job_status,
        caught.value.chunk_status,
    ) == ("job", chunk_id, job_status, chunk_status)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM generation_attempts").fetchone() == (0,)
        assert db.execute(
            "SELECT reserved_characters, reserved_requests FROM voiceover_jobs"
        ).fetchone() == (0, 0)
        assert db.execute(
            "SELECT status FROM voiceover_chunks WHERE job_id = ? AND chunk_id = ?",
            ("job", chunk_id),
        ).fetchone() == (chunk_status,)
