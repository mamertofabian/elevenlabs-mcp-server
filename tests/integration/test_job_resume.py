from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
from elevenlabs_mcp.database import JobStore
from elevenlabs_mcp.planner import ScriptPlanner

NOW = datetime(2026, 9, 21, tzinfo=UTC)


async def _store(path: Path, outcome: str = "failed"):
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
    chunk_id = plan.requests[0].chunk_id
    await store.reserve_attempt("job", chunk_id, "attempt", NOW)
    await store.mark_attempt_dispatched("attempt", NOW)
    if outcome == "failed":
        await store.record_attempt_failure("attempt", "failed", NOW)
    else:
        await store.record_attempt_failure("attempt", "unknown", NOW)
    return store, chunk_id


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
async def test_resume_requeues_with_new_ceiling_without_refunding_or_dispatch(tmp_path):
    store, chunk_id = await _store(tmp_path / "state.db")
    result = await store.request_resume("ws", "job", 0, "resume", 50, 5, NOW)
    assert (result.status, result.revision, result.replayed, result.warnings) == (
        "queued",
        1,
        False,
        (),
    )
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT cancel_requested,reserved_characters,reserved_requests,max_total_characters,max_total_requests FROM voiceover_jobs"
        ).fetchone() == (0, 5, 1, 50, 5)
        assert db.execute("SELECT status FROM voiceover_chunks").fetchone() == (
            "pending",
        )
        assert db.execute(
            "SELECT dispatch_state FROM generation_attempts"
        ).fetchall() == [("failed",)]
    await store.reserve_attempt("job", chunk_id, "new-attempt", NOW)
    await store.mark_attempt_dispatched("new-attempt", NOW)


@pytest.mark.asyncio
async def test_unknown_requires_ack_and_retains_charge_warning_and_evidence(tmp_path):
    from elevenlabs_mcp.database import UncertainAttemptError

    store, _ = await _store(tmp_path / "state.db", "unknown")
    before = _snapshot(store)
    with pytest.raises(UncertainAttemptError):
        await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    assert _snapshot(store) == before
    result = await store.request_resume(
        "ws", "job", 0, "resume", 100, 10, NOW, retry_uncertain=True
    )
    assert any("duplicate" in warning.lower() for warning in result.warnings)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT dispatch_state,outcome_unknown FROM generation_attempts"
        ).fetchone() == ("unknown", 1)


@pytest.mark.asyncio
async def test_replay_precedes_revision_and_conflicting_budget_fails(tmp_path):
    from elevenlabs_mcp.database import IdempotencyConflictError, RevisionConflictError

    store, _ = await _store(tmp_path / "state.db")
    result = await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    before = _snapshot(store)
    replay = await JobStore(store.db_path).request_resume(
        "ws", "job", 0, "resume", 100, 10, NOW
    )
    assert replay.model_copy(update={"replayed": False}) == result
    with pytest.raises(IdempotencyConflictError):
        await store.request_resume("ws", "job", 0, "resume", 101, 10, NOW)
    with pytest.raises(RevisionConflictError):
        await store.request_resume("ws", "job", 0, "other", 100, 10, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("characters,requests", [(4, 10), (100, 1)])
async def test_budget_cannot_fall_below_reserved_exposure(
    tmp_path, characters, requests
):
    store, _ = await _store(tmp_path / "state.db")
    from elevenlabs_mcp.database import BudgetExceededError

    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_jobs SET reserved_requests=2")
    before = _snapshot(store)
    with pytest.raises(BudgetExceededError):
        await store.request_resume("ws", "job", 0, "resume", characters, requests, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE voiceover_jobs SET status='running'",
        "UPDATE voiceover_chunks SET status='generating'",
        "UPDATE generation_attempts SET dispatch_state='dispatched'",
    ],
)
async def test_unreconciled_work_cannot_resume(tmp_path, mutation):
    from elevenlabs_mcp.database import JobBusyError

    store, _ = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(mutation)
    before = _snapshot(store)
    with pytest.raises(JobBusyError):
        await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_successful_chunks_require_artifact_verification(tmp_path):
    from elevenlabs_mcp.database import ArtifactVerificationRequiredError

    store, _ = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_chunks SET status='succeeded'")
    before = _snapshot(store)
    with pytest.raises(ArtifactVerificationRequiredError):
        await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_cancelled_job_resumes_but_old_reservation_cannot_dispatch(tmp_path):
    from elevenlabs_mcp.database import AttemptStateError

    store, chunk_id = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_jobs SET status='cancelled',cancel_requested=1")
        db.execute("UPDATE voiceover_chunks SET status='cancelled'")
        db.execute(
            "UPDATE generation_attempts SET dispatch_state='reserved',ended_at=NULL"
        )
    await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    with pytest.raises(AttemptStateError):
        await store.mark_attempt_dispatched("attempt", NOW)
    await store.reserve_attempt("job", chunk_id, "fresh", NOW)


@pytest.mark.asyncio
async def test_completed_job_is_unchanged(tmp_path):
    store, _ = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE voiceover_jobs SET status='completed'")
    before = _snapshot(store)[:3]
    result = await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    assert result.status == "completed" and result.revision == 0
    assert _snapshot(store)[:3] == before


@pytest.mark.asyncio
async def test_concurrent_identical_resume_mutates_once(tmp_path):
    store, _ = await _store(tmp_path / "state.db")
    results = await asyncio.gather(
        *[
            JobStore(store.db_path).request_resume(
                "ws", "job", 0, "resume", 100, 10, NOW
            )
            for _ in range(2)
        ]
    )
    assert sorted(result.replayed for result in results) == [False, True]


@pytest.mark.asyncio
async def test_receipt_failure_rolls_back_resume(tmp_path):
    store, _ = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON operation_receipts BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    before = _snapshot(store)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        await store.request_resume("ws", "job", 0, "resume", 100, 10, NOW)
    assert _snapshot(store) == before
