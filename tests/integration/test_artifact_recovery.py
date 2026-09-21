from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elevenlabs_mcp.artifacts import (
    ArtifactIdentity,
    ArtifactPublisher,
    ArtifactVerifier,
)
from elevenlabs_mcp.audio import AudioVerifier
from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
from elevenlabs_mcp.database import JobStore
from elevenlabs_mcp.planner import ScriptPlanner
from elevenlabs_mcp.workspace import WorkspaceLock

NOW = datetime(2026, 9, 21, tzinfo=UTC)


@pytest.fixture
def mp3():
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-f",
            "mp3",
            "-codec:a",
            "libmp3lame",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout


async def _store(root: Path):
    store = JobStore(root / "state.db")
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
    request = plan.requests[0]
    await store.reserve_attempt("job", request.chunk_id, "attempt", NOW)
    await store.mark_attempt_dispatched("attempt", NOW)
    return store, ArtifactIdentity(
        job_id="job",
        chunk_id=request.chunk_id,
        attempt_id="attempt",
        generation_fingerprint=request.generation_fingerprint,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_output", [False, True])
async def test_restart_adopts_complete_audio_without_resynthesis_and_retains_history(
    tmp_path, mp3, separate_output
):
    from elevenlabs_mcp.recovery import ArtifactRecovery

    store, identity = await _store(tmp_path)
    output_root = tmp_path / "audio" if separate_output else tmp_path
    output_root.mkdir(exist_ok=True)
    ArtifactPublisher(output_root).publish(identity, [mp3])
    with WorkspaceLock(tmp_path) as owner:
        result = await ArtifactRecovery(
            store, AudioVerifier(ArtifactVerifier(output_root))
        ).recover(owner, NOW)
        repeated = await ArtifactRecovery(
            JobStore(store.db_path), AudioVerifier(ArtifactVerifier(output_root))
        ).recover(owner, NOW)
    assert len(result.adopted_artifact_ids) == 1
    assert result.failures == ()
    assert repeated.adopted_artifact_ids == ()
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT status,verified_chunks,verified_parts,reserved_requests,revision FROM voiceover_jobs"
        ).fetchone() == ("paused", 1, 1, 1, 0)
        assert db.execute(
            "SELECT dispatch_state,outcome_unknown FROM generation_attempts"
        ).fetchall() == [("unknown", 1)]
        assert db.execute(
            "SELECT status,successful_artifact_id FROM voiceover_chunks"
        ).fetchone() == ("succeeded", result.adopted_artifact_ids[0])
        assert db.execute("SELECT COUNT(*) FROM production_artifacts").fetchone() == (
            1,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing_marker", "corrupt", "invalid_audio"])
async def test_unverified_source_stays_unknown_with_visible_failure(
    tmp_path, mp3, damage
):
    from elevenlabs_mcp.recovery import ArtifactRecovery

    store, identity = await _store(tmp_path)
    published = ArtifactPublisher(tmp_path).publish(
        identity, [b"not audio" if damage == "invalid_audio" else mp3]
    )
    path = tmp_path / published.relative_path
    if damage == "missing_marker":
        path.with_suffix(".complete.json").unlink()
    elif damage == "corrupt":
        path.write_bytes(b"corrupt")
    with WorkspaceLock(tmp_path) as owner:
        result = await ArtifactRecovery(
            store, AudioVerifier(ArtifactVerifier(tmp_path))
        ).recover(owner, NOW)
    assert result.adopted_artifact_ids == ()
    assert result.failures[0].attempt_id == "attempt"
    assert result.failures[0].code == (
        "AUDIO_CHECK_FAILED" if damage == "invalid_audio" else "INTEGRITY_CHECK_FAILED"
    )
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT status FROM voiceover_chunks").fetchone() == (
            "unknown",
        )
        assert db.execute("SELECT COUNT(*) FROM production_artifacts").fetchone() == (
            0,
        )


@pytest.mark.asyncio
async def test_newer_attempt_prevents_adoption_of_stale_complete_source(tmp_path, mp3):
    from elevenlabs_mcp.recovery import ArtifactRecovery

    store, identity = await _store(tmp_path)
    ArtifactPublisher(tmp_path).publish(identity, [mp3])
    with WorkspaceLock(tmp_path) as owner:
        await store.reconcile_interrupted(owner, NOW)
        await store.request_resume(
            "ws", "job", 0, "resume", 100, 10, NOW, retry_uncertain=True
        )
        await store.reserve_attempt("job", identity.chunk_id, "new-attempt", NOW)
        await store.mark_attempt_dispatched("new-attempt", NOW)
        result = await ArtifactRecovery(
            store, AudioVerifier(ArtifactVerifier(tmp_path))
        ).recover(owner, NOW)
    assert result.adopted_artifact_ids == ()
    assert [failure.attempt_id for failure in result.failures] == ["new-attempt"]


@pytest.mark.asyncio
async def test_adoption_rechecks_state_after_verification(tmp_path, mp3):
    from elevenlabs_mcp.database import AttemptStateError

    store, identity = await _store(tmp_path)
    ArtifactPublisher(tmp_path).publish(identity, [mp3])
    verified = await asyncio.to_thread(
        AudioVerifier(ArtifactVerifier(tmp_path)).verify, identity
    )
    with WorkspaceLock(tmp_path) as owner:
        await store.reconcile_interrupted(owner, NOW)
        await store.request_resume(
            "ws", "job", 0, "resume", 100, 10, NOW, retry_uncertain=True
        )
        await store.reserve_attempt("job", identity.chunk_id, "new-attempt", NOW)
        with pytest.raises(AttemptStateError):
            await store.record_verified_artifact(
                "recovered", verified, NOW, recovery_owner=owner
            )
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM production_artifacts").fetchone() == (
            0,
        )


@pytest.mark.asyncio
async def test_adoption_requires_current_database_owner(tmp_path, mp3):
    from elevenlabs_mcp.workspace import WorkspaceOwnershipError

    store, identity = await _store(tmp_path)
    ArtifactPublisher(tmp_path).publish(identity, [mp3])
    verified = await asyncio.to_thread(
        AudioVerifier(ArtifactVerifier(tmp_path)).verify, identity
    )
    with pytest.raises(WorkspaceOwnershipError):
        await store.record_verified_artifact(
            "recovered", verified, NOW, recovery_owner=WorkspaceLock(tmp_path)
        )
