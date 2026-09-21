from __future__ import annotations

import select
import sqlite3
import subprocess
import sys
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
        PlanningLimits(max_text_characters=3),
    )
    await store.create_job("ws", "job", plan, "submit", 100, 10, NOW)
    for i, request in enumerate(plan.requests):
        await store.reserve_attempt("job", request.chunk_id, f"attempt{i}", NOW)
    await store.mark_attempt_dispatched("attempt0", NOW)
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
async def test_restart_pauses_and_preserves_exposure_without_replaying(tmp_path):
    from elevenlabs_mcp.workspace import WorkspaceLock

    store = await _store(tmp_path / "state.db")
    with WorkspaceLock(tmp_path) as owner:
        result = await store.reconcile_interrupted(owner, NOW + timedelta(seconds=1))
    assert result.paused_job_ids == ("job",)
    assert result.uncertain_attempt_ids == ("attempt0",)
    assert result.abandoned_reservation_ids == ("attempt1",)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT status,reason,reserved_characters,reserved_requests,revision FROM voiceover_jobs"
        ).fetchone() == ("paused", "UPSTREAM_OUTCOME_UNKNOWN", 5, 2, 0)
        assert db.execute(
            "SELECT dispatch_state,outcome_unknown FROM generation_attempts ORDER BY attempt_id"
        ).fetchall() == [("unknown", 1), ("cancelled", 0)]
        assert db.execute(
            "SELECT status FROM voiceover_chunks ORDER BY chunk_index"
        ).fetchall() == [("unknown",), ("pending",)]
    before = _snapshot(store)
    with WorkspaceLock(tmp_path) as owner:
        repeated = await store.reconcile_interrupted(owner, NOW + timedelta(seconds=2))
    assert repeated.paused_job_ids == ()
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_reconciliation_requires_live_workspace_ownership(tmp_path):
    from elevenlabs_mcp.workspace import WorkspaceLock, WorkspaceOwnershipError

    store = await _store(tmp_path / "state.db")
    owner = WorkspaceLock(tmp_path)
    before = _snapshot(store)
    with pytest.raises(WorkspaceOwnershipError):
        await store.reconcile_interrupted(owner, NOW)
    with owner:
        pass
    with pytest.raises(WorkspaceOwnershipError):
        await store.reconcile_interrupted(owner, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "assembling"])
async def test_abandoned_job_without_active_attempts_is_paused(tmp_path, status):
    from elevenlabs_mcp.workspace import WorkspaceLock

    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE generation_attempts SET dispatch_state='failed'")
        db.execute("UPDATE voiceover_chunks SET status='failed'")
        db.execute("UPDATE voiceover_jobs SET status=?,cancel_requested=1", (status,))
    with WorkspaceLock(tmp_path) as owner:
        await store.reconcile_interrupted(owner, NOW)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT status,reason,cancel_requested FROM voiceover_jobs"
        ).fetchone() == ("paused", "PROCESS_RESTARTED", 1)


@pytest.mark.asyncio
async def test_restart_transaction_rolls_back_on_write_failure(tmp_path):
    from elevenlabs_mcp.workspace import WorkspaceLock

    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_update BEFORE UPDATE ON voiceover_jobs BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    before = _snapshot(store)
    with WorkspaceLock(tmp_path) as owner, pytest.raises(sqlite3.IntegrityError):
        await store.reconcile_interrupted(owner, NOW)
    assert _snapshot(store) == before


def test_second_owner_is_refused_and_lock_survives_process_exit(tmp_path):
    from elevenlabs_mcp.workspace import WorkspaceBusyError, WorkspaceLock

    # Real child ownership; readiness is read from stdout rather than timed sleeps.
    program = "from pathlib import Path; import sys; from elevenlabs_mcp.workspace import WorkspaceLock; owner=WorkspaceLock(Path(sys.argv[1])); owner.__enter__(); print('locked',flush=True); sys.stdin.read()"
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0], (
            "Child lock acquisition timed out"
        )
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(WorkspaceBusyError), WorkspaceLock(tmp_path):
            pass
    finally:
        child.kill()
        child.communicate(timeout=10)
    with WorkspaceLock(tmp_path) as owner:
        owner.require_held()


def test_symlink_lock_file_cannot_touch_external_file(tmp_path):
    from elevenlabs_mcp.workspace import WorkspaceLock

    external = tmp_path / "external"
    external.write_text("untouched")
    (tmp_path / ".elevenlabs-workspace.lock").symlink_to(external)
    with pytest.raises(OSError), WorkspaceLock(tmp_path):
        pass
    assert external.read_text() == "untouched"


@pytest.mark.asyncio
async def test_unrelated_workspace_lock_cannot_reconcile_database(tmp_path):
    from elevenlabs_mcp.workspace import WorkspaceLock, WorkspaceOwnershipError

    store = await _store(tmp_path / "state.db")
    other = tmp_path / "other"
    other.mkdir()
    before = _snapshot(store)
    with WorkspaceLock(other) as owner, pytest.raises(WorkspaceOwnershipError):
        await store.reconcile_interrupted(owner, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE voiceover_jobs SET status='completed'",
        "UPDATE voiceover_jobs SET deleted_at='deleted'",
    ],
)
async def test_completed_or_deleted_jobs_are_not_rewritten(tmp_path, mutation):
    from elevenlabs_mcp.workspace import WorkspaceLock

    store = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(mutation)
    before = _snapshot(store)
    with WorkspaceLock(tmp_path) as owner:
        result = await store.reconcile_interrupted(owner, NOW)
    assert result.paused_job_ids == ()
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
async def test_database_alias_cannot_bypass_existing_owner(tmp_path, alias_kind):
    import os

    from elevenlabs_mcp.workspace import WorkspaceLock, WorkspaceOwnershipError

    original = tmp_path / "original"
    alias = tmp_path / "alias"
    alias.mkdir()
    store = await _store(original / "state.db")
    if alias_kind == "symlink":
        (alias / "state.db").symlink_to(original / "state.db")
    else:
        os.link(original / "state.db", alias / "state.db")
    before = _snapshot(store)
    with (
        WorkspaceLock(original),
        WorkspaceLock(alias) as wrong_owner,
        pytest.raises(WorkspaceOwnershipError),
    ):
        await JobStore(alias / "state.db").reconcile_interrupted(wrong_owner, NOW)
    assert _snapshot(store) == before
