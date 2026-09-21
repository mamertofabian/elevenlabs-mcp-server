from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elevenlabs_mcp.artifacts import ArtifactIdentity, ArtifactIntegrityResult
from elevenlabs_mcp.audio import AudioVerificationResult
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
    results = []
    for i, request in enumerate(plan.requests):
        attempt_id = f"attempt{i}"
        await store.reserve_attempt("job", request.chunk_id, attempt_id, NOW)
        await store.mark_attempt_dispatched(attempt_id, NOW)
        identity = ArtifactIdentity(
            job_id="job",
            chunk_id=request.chunk_id,
            attempt_id=attempt_id,
            generation_fingerprint=request.generation_fingerprint,
        )
        results.append(
            AudioVerificationResult(
                integrity=ArtifactIntegrityResult(
                    identity=identity,
                    relative_path=f"jobs/job/chunks/{request.chunk_id}/{attempt_id}.mp3",
                    sha256="sha256:" + "a" * 64,
                    byte_size=123,
                ),
                duration_ms=100,
            )
        )
    return store, results


def _snapshot(store):
    with sqlite3.connect(store.db_path) as db:
        return tuple(
            db.execute(f"SELECT * FROM {table}").fetchall()
            for table in (
                "voiceover_jobs",
                "voiceover_chunks",
                "generation_attempts",
                "production_artifacts",
            )
        )


@pytest.mark.asyncio
async def test_atomic_success_counts_original_parts_only_when_all_fragments_succeed(
    tmp_path,
):
    store, results = await _store(tmp_path / "state.db")
    recorded = await store.record_verified_artifact("artifact0", results[0], NOW)
    assert recorded.artifact_id == "artifact0" and recorded.replayed is False
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT verified_chunks,verified_parts,status,reserved_characters,reserved_requests,revision FROM voiceover_jobs"
        ).fetchone() == (1, 0, "running", 5, 2, 0)
        assert db.execute(
            "SELECT status,successful_artifact_id FROM voiceover_chunks WHERE chunk_index=0"
        ).fetchone() == ("succeeded", "artifact0")
        assert db.execute(
            "SELECT dispatch_state,ended_at,sanitized_outcome FROM generation_attempts WHERE attempt_id='attempt0'"
        ).fetchone() == ("succeeded", NOW.isoformat(), "AUDIO_VERIFIED")
        assert db.execute(
            "SELECT mime_type,codec,duration_ms,complete FROM production_artifacts"
        ).fetchone() == ("audio/mpeg", "mp3", 100, 1)
    await store.record_verified_artifact("artifact1", results[1], NOW)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT verified_chunks,verified_parts,status FROM voiceover_jobs"
        ).fetchone() == (2, 1, "running")


@pytest.mark.asyncio
async def test_concurrent_identical_recording_has_one_insert_and_safe_replay(tmp_path):
    store, results = await _store(tmp_path / "state.db")
    recorded = await asyncio.gather(
        *[
            JobStore(store.db_path).record_verified_artifact(
                "artifact", results[0], NOW
            )
            for _ in range(2)
        ]
    )
    assert sorted(result.replayed for result in recorded) == [False, True]
    before = _snapshot(store)
    replay = await store.record_verified_artifact("artifact", results[0], NOW)
    assert replay.replayed is True and _snapshot(store) == before


@pytest.mark.asyncio
async def test_conflicting_artifact_id_or_relabelled_success_is_rejected(tmp_path):
    from elevenlabs_mcp.database import ArtifactConflictError, AttemptStateError

    store, results = await _store(tmp_path / "state.db")
    await store.record_verified_artifact("artifact", results[0], NOW)
    before = _snapshot(store)
    with pytest.raises(ArtifactConflictError):
        await store.record_verified_artifact("artifact", results[1], NOW)
    with pytest.raises(AttemptStateError):
        await store.record_verified_artifact("other-artifact", results[0], NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["fingerprint", "owner", "path", "invalid_size"])
async def test_verification_metadata_cannot_escape_ownership_or_immutable_plan(
    tmp_path, damage
):
    from elevenlabs_mcp.database import ArtifactConflictError

    store, results = await _store(tmp_path / "state.db")
    data = results[0].model_dump()
    if damage == "fingerprint":
        data["integrity"]["identity"]["generation_fingerprint"] = "sha256:" + "b" * 64
    elif damage == "owner":
        data["integrity"]["identity"]["job_id"] = "other"
    elif damage == "path":
        data["integrity"]["relative_path"] = "../private.mp3"
    else:
        data["integrity"]["byte_size"] = 0
    data["integrity"]["identity"] = ArtifactIdentity.model_validate(
        data["integrity"]["identity"]
    )
    if damage == "owner":
        data["integrity"]["relative_path"] = results[0].integrity.relative_path.replace(
            "jobs/job/", "jobs/other/"
        )
    # model_construct demonstrates why the store must validate even typed inputs.
    forged = results[0].model_copy(
        update={
            "integrity": ArtifactIntegrityResult.model_construct(**data["integrity"])
        }
    )
    before = _snapshot(store)
    with pytest.raises((ValueError, ArtifactConflictError)):
        await store.record_verified_artifact("artifact", forged, NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_cancel_request_allows_inflight_result_without_erasing_uncertainty(
    tmp_path,
):
    store, results = await _store(tmp_path / "state.db")
    await store.request_cancel("ws", "job", 0, "cancel", NOW)
    await store.record_attempt_failure("attempt1", "unknown", NOW)
    await store.record_verified_artifact("artifact", results[0], NOW)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT status,reason,cancel_requested,revision FROM voiceover_jobs"
        ).fetchone() == ("paused", "UPSTREAM_OUTCOME_UNKNOWN", 1, 1)


@pytest.mark.asyncio
async def test_transaction_failure_rolls_back_artifact_attempt_chunk_and_counts(
    tmp_path,
):
    store, results = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_progress BEFORE UPDATE ON voiceover_jobs BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    before = _snapshot(store)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        await store.record_verified_artifact("artifact", results[0], NOW)
    assert _snapshot(store) == before


@pytest.mark.asyncio
async def test_real_verified_audio_can_be_recorded(tmp_path):
    import hashlib
    import json
    import subprocess

    from elevenlabs_mcp.artifacts import ArtifactVerifier, CompletionRecord
    from elevenlabs_mcp.audio import AudioVerifier

    store, results = await _store(tmp_path / "state.db")
    identity = results[0].integrity.identity
    source = tmp_path / results[0].integrity.relative_path
    source.parent.mkdir(parents=True)
    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-codec:a",
            "libmp3lame",
            str(source),
        ],
        check=True,
        timeout=10,
        capture_output=True,
    )
    payload = source.read_bytes()
    marker = CompletionRecord(
        schema_version="1",
        identity=identity,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )
    source.with_suffix(".complete.json").write_text(json.dumps(marker.model_dump()))
    verified = await asyncio.to_thread(
        AudioVerifier(ArtifactVerifier(tmp_path)).verify, identity
    )
    await store.record_verified_artifact("real-artifact", verified, NOW)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "SELECT sha256,byte_size,duration_ms FROM production_artifacts"
        ).fetchone() == (marker.sha256, len(payload), verified.duration_ms)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE generation_attempts SET dispatch_state='unknown',outcome_unknown=1",
        "UPDATE voiceover_jobs SET deleted_at='deleted'",
        "UPDATE voiceover_jobs SET status='completed'",
    ],
)
async def test_unreconciled_or_ineligible_attempt_is_not_adopted(tmp_path, mutation):
    from elevenlabs_mcp.database import AttemptStateError

    store, results = await _store(tmp_path / "state.db")
    with sqlite3.connect(store.db_path) as db:
        db.execute(mutation)
    before = _snapshot(store)
    with pytest.raises(AttemptStateError):
        await store.record_verified_artifact("artifact", results[0], NOW)
    assert _snapshot(store) == before
