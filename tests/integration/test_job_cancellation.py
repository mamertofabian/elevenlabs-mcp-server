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


async def _store(path: Path):
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
    await store.create_job("ws", "job", plan, "submit", 100, 10, NOW)
    await store.reserve_attempt("job", plan.requests[0].chunk_id, "attempt", NOW)
    return store


def _snapshot(store):
    with sqlite3.connect(store.db_path) as db:
        return tuple(
            db.execute(f"SELECT * FROM {table}").fetchall()
            for table in (
                "voiceover_jobs",
                "voiceover_chunks",
                "generation_attempts",
                "operation_receipts",
            )
        )


@pytest.mark.asyncio
async def test_cancel_blocks_dispatch_without_claiming_upstream_stopped(tmp_path):
    from elevenlabs_mcp.database import AttemptStateError

    store = await _store(tmp_path / "state.db")
    result = await store.request_cancel("ws", "job", 0, "cancel", NOW)
    assert (
        result.job_id,
        result.revision,
        result.status,
        result.cancel_requested,
        result.replayed,
    ) == ("job", 1, "queued", True, False)
    with pytest.raises(AttemptStateError):
        await store.mark_attempt_dispatched("attempt", NOW)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT reserved_characters,reserved_requests FROM voiceover_jobs"
        ).fetchone() == (5, 1)
        assert db.execute("SELECT status FROM voiceover_chunks").fetchone() == (
            "pending",
        )
        assert db.execute(
            "SELECT dispatch_state FROM generation_attempts"
        ).fetchone() == ("reserved",)


@pytest.mark.asyncio
async def test_identical_receipt_replays_original_ack_before_revision_check(tmp_path):
    store = await _store(tmp_path / "state.db")
    original = await store.request_cancel("ws", "job", 0, "cancel", NOW)
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_jobs SET revision=4, status='cancelled'")
    before = _snapshot(store)
    replay = await JobStore(store.db_path).request_cancel(
        "ws", "job", 0, "cancel", NOW + timedelta(seconds=1)
    )
    assert replay.model_copy(update={"replayed": False}) == original
    assert replay.replayed is True
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_stale_revision_and_conflicting_key_leave_state_unchanged(tmp_path):
    from elevenlabs_mcp.database import IdempotencyConflictError, RevisionConflictError

    store = await _store(tmp_path / "state.db")
    await store.request_cancel("ws", "job", 0, "cancel", NOW)
    before = _snapshot(store)
    with pytest.raises(RevisionConflictError) as caught:
        await store.request_cancel("ws", "job", 0, "new-key", NOW)
    assert (caught.value.expected_revision, caught.value.actual_revision) == (0, 1)
    with pytest.raises(IdempotencyConflictError):
        await store.request_cancel("ws", "job", 1, "cancel", NOW)
    with pytest.raises(IdempotencyConflictError):
        await store.request_cancel("ws", "other-job", 0, "cancel", NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("same_key", [True, False])
async def test_concurrent_cancellation_has_one_mutation(tmp_path, same_key):
    from elevenlabs_mcp.database import RevisionConflictError

    store = await _store(tmp_path / "state.db")
    results = await asyncio.gather(
        store.request_cancel("ws", "job", 0, "cancel", NOW),
        JobStore(store.db_path).request_cancel(
            "ws", "job", 0, "cancel" if same_key else "other", NOW
        ),
        return_exceptions=True,
    )
    if same_key:
        assert sorted(
            result.replayed
            for result in results
            if not isinstance(result, BaseException)
        ) == [False, True]
    else:
        assert sum(isinstance(result, RevisionConflictError) for result in results) == 1
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT revision,cancel_requested FROM voiceover_jobs"
        ).fetchone() == (1, 1)
        assert db.execute(
            "SELECT COUNT(*) FROM operation_receipts WHERE operation='cancel'"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_cancelling_inflight_attempt_preserves_later_unknown_result(tmp_path):
    store = await _store(tmp_path / "state.db")
    await store.mark_attempt_dispatched("attempt", NOW)
    ack = await store.request_cancel("ws", "job", 0, "cancel", NOW)
    assert ack.status == "running" and ack.cancel_requested is True
    await store.record_attempt_failure("attempt", "unknown", NOW)
    repeated = await store.request_cancel("ws", "job", 1, "again", NOW)
    assert (repeated.status, repeated.reason, repeated.revision) == (
        "paused",
        "UPSTREAM_OUTCOME_UNKNOWN",
        1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "cancelled"])
async def test_terminal_job_cancellation_does_not_rewrite_job_or_artifacts(
    tmp_path, status
):
    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_jobs SET status=?,revision=3", (status,))
    before = _snapshot(store)[:3]
    result = await store.request_cancel("ws", "job", 3, "cancel", NOW)
    assert result.status == status and result.revision == 3
    assert _snapshot(store)[:3] == before


@pytest.mark.asyncio
async def test_missing_deleted_and_invalid_requests_do_not_leave_receipts(tmp_path):
    from elevenlabs_mcp.database import JobNotFoundError

    store = await _store(tmp_path / "state.db")
    before = _snapshot(store)
    with pytest.raises(JobNotFoundError):
        await store.request_cancel("ws", "missing", 0, "cancel", NOW)
    for revision, key, timestamp in [
        (-1, "cancel", NOW),
        (True, "cancel", NOW),
        (0, "", NOW),
        (0, "cancel", NOW.replace(tzinfo=None)),
    ]:
        with pytest.raises(ValueError):
            await store.request_cancel("ws", "job", revision, key, timestamp)
    assert _snapshot(store) == before
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_jobs SET deleted_at='deleted'")
    before = _snapshot(store)
    with pytest.raises(JobNotFoundError):
        await store.request_cancel("ws", "job", 0, "cancel", NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_receipt_failure_rolls_back_cancellation(tmp_path):
    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON operation_receipts BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    before = _snapshot(store)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        await store.request_cancel("ws", "job", 0, "cancel", NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_cancel_prevents_new_budget_reservations(tmp_path):
    from elevenlabs_mcp.database import ReservationStateError

    store = await _store(tmp_path / "state.db")
    await store.request_cancel("ws", "job", 0, "cancel", NOW)
    with sqlite3.connect(store.db_path) as db:
        chunk_id = db.execute("SELECT chunk_id FROM voiceover_chunks").fetchone()[0]
    before = _snapshot(store)
    with pytest.raises(ReservationStateError):
        await store.reserve_attempt("job", chunk_id, "new-attempt", NOW)
    # Re-reading an existing reservation is still harmless.
    replay = await store.reserve_attempt("job", chunk_id, "attempt", NOW)
    assert replay.replayed is True
    assert _snapshot(store) == before
