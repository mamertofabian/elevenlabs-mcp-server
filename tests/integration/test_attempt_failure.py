from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
from elevenlabs_mcp.database import JobStore
from elevenlabs_mcp.planner import ScriptPlanner

NOW = datetime(2026, 9, 21, tzinfo=UTC)
END = NOW + timedelta(seconds=5)


async def _store(path: Path, count: int = 1):
    store = JobStore(path)
    await store.initialize()
    script = Script.model_validate(
        {
            "script_version": "1",
            "cast": {"actor": {"voice_id": "voice"}},
            "scenes": [
                {
                    "id": "scene",
                    "parts": [
                        {"id": f"part{i}", "actor": "actor", "text": "hello"}
                        for i in range(count)
                    ],
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
    for i, request in enumerate(plan.requests):
        await store.reserve_attempt("job", request.chunk_id, f"attempt{i}", NOW)
        await store.mark_attempt_dispatched(f"attempt{i}", NOW)
    return store


def _snapshot(store):
    with sqlite3.connect(store.db_path) as db:
        return tuple(
            db.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("voiceover_jobs", "voiceover_chunks", "generation_attempts")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, status, reason",
    [
        ("failed", "failed", "GENERATION_FAILED"),
        ("unknown", "paused", "UPSTREAM_OUTCOME_UNKNOWN"),
    ],
)
async def test_failure_is_durable_and_keeps_budget(tmp_path, outcome, status, reason):
    store = await _store(tmp_path / "state.db")
    result = await store.record_attempt_failure("attempt0", outcome, END, "provider-id")
    assert (result.attempt_id, result.job_id, result.outcome, result.replayed) == (
        "attempt0",
        "job",
        outcome,
        False,
    )
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT status, reason, reserved_characters, reserved_requests, revision FROM voiceover_jobs"
        ).fetchone() == (status, reason, 5, 1, 0)
        assert db.execute(
            "SELECT status, latest_error FROM voiceover_chunks"
        ).fetchone() == (outcome, reason)
        assert db.execute(
            "SELECT dispatch_state, ended_at, provider_request_id, sanitized_outcome, outcome_unknown FROM generation_attempts"
        ).fetchone() == (
            outcome,
            END.isoformat(),
            "provider-id",
            reason,
            int(outcome == "unknown"),
        )


@pytest.mark.asyncio
async def test_repeated_result_is_idempotent_but_conflicts_fail(tmp_path):
    from elevenlabs_mcp.database import AttemptOutcomeConflictError

    store = await _store(tmp_path / "state.db")
    await store.record_attempt_failure("attempt0", "unknown", END, "provider-id")
    before = _snapshot(store)
    replay = await JobStore(store.db_path).record_attempt_failure(
        "attempt0", "unknown", END + timedelta(seconds=1), "provider-id"
    )
    assert replay.replayed is True
    conflicts: list[tuple[Literal["failed", "unknown"], str]] = [
        ("failed", "provider-id"),
        ("unknown", "other"),
    ]
    for outcome, provider_id in conflicts:
        with pytest.raises(AttemptOutcomeConflictError):
            await store.record_attempt_failure("attempt0", outcome, END, provider_id)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_concurrent_identical_results_commit_once(tmp_path):
    store = await _store(tmp_path / "state.db")
    results = await asyncio.gather(
        *[
            JobStore(store.db_path).record_attempt_failure("attempt0", "unknown", END)
            for _ in range(2)
        ]
    )
    assert sorted(result.replayed for result in results) == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcomes", [("failed", "unknown"), ("unknown", "failed")])
async def test_unknown_chunk_keeps_job_paused_when_other_attempt_finishes(
    tmp_path, outcomes
):
    store = await _store(tmp_path / "state.db", count=2)
    for i, outcome in enumerate(outcomes):
        await store.record_attempt_failure(f"attempt{i}", outcome, END)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT status, reason, reserved_requests FROM voiceover_jobs"
        ).fetchone() == ("paused", "UPSTREAM_OUTCOME_UNKNOWN", 2)


@pytest.mark.asyncio
async def test_invalid_result_and_time_do_not_mutate(tmp_path):
    from elevenlabs_mcp.database import AttemptNotFoundError

    store = await _store(tmp_path / "state.db")
    before = _snapshot(store)
    with pytest.raises(AttemptNotFoundError):
        await store.record_attempt_failure("missing", "unknown", END)
    for outcome, time, provider in [
        ("raw private provider body", END, None),
        ("unknown", NOW - timedelta(seconds=1), None),
        ("unknown", END.replace(tzinfo=None), None),
        ("unknown", END, "x" * 129),
    ]:
        with pytest.raises(ValueError):
            await store.record_attempt_failure("attempt0", outcome, time, provider)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE generation_attempts SET dispatch_state='reserved'",
        "UPDATE voiceover_chunks SET status='succeeded'",
        "UPDATE voiceover_jobs SET status='completed'",
        "UPDATE voiceover_jobs SET deleted_at='deleted'",
    ],
)
async def test_invalid_state_cannot_be_finalized(tmp_path, mutation):
    from elevenlabs_mcp.database import AttemptStateError

    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(mutation)
    before = _snapshot(store)
    with pytest.raises(AttemptStateError):
        await store.record_attempt_failure("attempt0", "unknown", END)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_failure_transaction_rolls_back_all_records(tmp_path):
    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_update BEFORE UPDATE ON voiceover_jobs BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    before = _snapshot(store)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        await store.record_attempt_failure("attempt0", "unknown", END)
    assert _snapshot(store) == before
