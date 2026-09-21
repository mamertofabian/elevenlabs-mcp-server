from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
from elevenlabs_mcp.database import JobStore
from elevenlabs_mcp.planner import ScriptPlanner

NOW = datetime(2026, 9, 21, tzinfo=UTC)


async def _reserved_store(path: Path):
    store = JobStore(path)
    await store.initialize()
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
    plan = ScriptPlanner().plan(
        script,
        VoiceoverOptions(engine="tts", model_id="eleven_multilingual_v2"),
        PlanningLimits(),
    )
    await store.create_job("workspace", "job", plan, "submit", 100, 10, NOW)
    chunk_id = plan.requests[0].chunk_id
    await store.reserve_attempt("job", chunk_id, "attempt", NOW)
    return store, chunk_id


def _state(path: Path):
    with sqlite3.connect(path) as db:
        return (
            db.execute(
                "SELECT status, reserved_characters, reserved_requests, revision FROM voiceover_jobs"
            ).fetchone(),
            db.execute("SELECT status FROM voiceover_chunks").fetchone(),
            db.execute(
                "SELECT attempt_id, dispatch_state FROM generation_attempts ORDER BY attempt_id"
            ).fetchall(),
        )


@pytest.mark.asyncio
async def test_dispatch_commits_before_return_without_reserving_again(tmp_path):
    store, chunk_id = await _reserved_store(tmp_path / "state.db")
    result = await store.mark_attempt_dispatched("attempt", NOW + timedelta(seconds=1))
    assert (result.attempt_id, result.job_id, result.chunk_id) == (
        "attempt",
        "job",
        chunk_id,
    )
    assert _state(Path(store.db_path)) == (
        ("running", 5, 1, 0),
        ("generating",),
        [("attempt", "dispatched")],
    )
    # A new connection sees the durable transition before a caller can send HTTP.
    with sqlite3.connect(store.db_path) as db:
        assert (
            db.execute("SELECT updated_at FROM voiceover_jobs").fetchone()[0]
            == (NOW + timedelta(seconds=1)).isoformat()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("different_attempts", [False, True])
async def test_concurrent_dispatch_has_exactly_one_winner(tmp_path, different_attempts):
    from elevenlabs_mcp.database import AttemptStateError

    store, chunk_id = await _reserved_store(tmp_path / "state.db")
    second = "attempt"
    if different_attempts:
        second = "second"
        await store.reserve_attempt("job", chunk_id, second, NOW)
    results = await asyncio.gather(
        store.mark_attempt_dispatched("attempt", NOW),
        JobStore(store.db_path).mark_attempt_dispatched(second, NOW),
        return_exceptions=True,
    )
    assert sum(not isinstance(item, BaseException) for item in results) == 1
    assert sum(isinstance(item, AttemptStateError) for item in results) == 1
    assert (
        sum(state == "dispatched" for _, state in _state(Path(store.db_path))[2]) == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE voiceover_jobs SET cancel_requested=1",
        "UPDATE voiceover_jobs SET status='paused'",
        "UPDATE voiceover_jobs SET deleted_at='deleted'",
        "UPDATE voiceover_chunks SET status='succeeded'",
    ],
)
async def test_ineligible_work_cannot_dispatch(tmp_path, mutation):
    from elevenlabs_mcp.database import AttemptStateError

    store, _ = await _reserved_store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(mutation)
    before = _state(Path(store.db_path))
    with pytest.raises(AttemptStateError):
        await store.mark_attempt_dispatched("attempt", NOW)
    assert _state(Path(store.db_path)) == before


@pytest.mark.asyncio
async def test_missing_attempt_and_invalid_time_do_not_mutate(tmp_path):
    from elevenlabs_mcp.database import AttemptNotFoundError

    store, _ = await _reserved_store(tmp_path / "state.db")
    before = _state(Path(store.db_path))
    with pytest.raises(AttemptNotFoundError):
        await store.mark_attempt_dispatched("missing", NOW)
    for invalid in (NOW.replace(tzinfo=None), NOW - timedelta(seconds=1)):
        with pytest.raises(ValueError):
            await store.mark_attempt_dispatched("attempt", invalid)
    assert _state(Path(store.db_path)) == before


@pytest.mark.asyncio
async def test_failed_dispatch_transaction_rolls_back_all_records(tmp_path):
    store, _ = await _reserved_store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_running BEFORE UPDATE ON voiceover_jobs BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    before = _state(Path(store.db_path))
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        await store.mark_attempt_dispatched("attempt", NOW)
    assert _state(Path(store.db_path)) == before
