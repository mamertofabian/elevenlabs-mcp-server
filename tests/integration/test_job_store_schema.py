from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest


def _columns(path: Path, table: str) -> tuple[str, ...]:
    with sqlite3.connect(path) as db:
        return tuple(row[1] for row in db.execute(f"PRAGMA table_info({table})"))


def _insert_job(db: sqlite3.Connection, job_id: str) -> None:
    db.execute(
        """INSERT INTO voiceover_jobs (
        job_id,schema_version,plan_version,plan_hash,normalized_script_json,
        options_json,plan_json,status,reason,revision,created_at,updated_at,
        total_parts,verified_parts,total_chunks,verified_chunks,
        max_total_characters,max_total_requests,reserved_characters,
        reserved_requests,cancel_requested,source,deleted_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            job_id,
            "1",
            "1",
            "sha256:" + "0" * 64,
            "{}",
            "{}",
            "{}",
            "queued",
            None,
            0,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            "native",
            None,
        ),
    )


def _insert_chunk(db: sqlite3.Connection, chunk_id: str, job_id: str) -> None:
    db.execute(
        """INSERT INTO voiceover_chunks (
        chunk_id,job_id,chunk_index,generation_fingerprint,status,character_count,
        voice_ids_json,pause_after_ms,source_spans_json,request_json)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            chunk_id,
            job_id,
            0,
            "sha256:" + "1" * 64,
            "pending",
            1,
            '["voice"]',
            0,
            "[]",
            "{}",
        ),
    )


@pytest.mark.asyncio
async def test_initialize_creates_complete_versioned_additive_schema(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    path = tmp_path / "voiceover_history.db"
    store = JobStore(path)
    await store.initialize()
    assert await store.get_schema_version() == 1
    expected = {
        "voiceover_jobs": (
            "job_id",
            "schema_version",
            "plan_version",
            "plan_hash",
            "normalized_script_json",
            "options_json",
            "plan_json",
            "status",
            "reason",
            "revision",
            "created_at",
            "updated_at",
            "total_parts",
            "verified_parts",
            "total_chunks",
            "verified_chunks",
            "max_total_characters",
            "max_total_requests",
            "reserved_characters",
            "reserved_requests",
            "cancel_requested",
            "source",
            "deleted_at",
        ),
        "voiceover_chunks": (
            "chunk_id",
            "job_id",
            "chunk_index",
            "generation_fingerprint",
            "status",
            "character_count",
            "voice_ids_json",
            "pause_after_ms",
            "source_spans_json",
            "request_json",
            "successful_artifact_id",
            "latest_error",
            "generation_metadata_json",
        ),
        "generation_attempts": (
            "attempt_id",
            "job_id",
            "chunk_id",
            "dispatch_state",
            "reserved_characters",
            "started_at",
            "ended_at",
            "provider_request_id",
            "sanitized_outcome",
            "outcome_unknown",
        ),
        "production_artifacts": (
            "artifact_id",
            "job_id",
            "chunk_id",
            "attempt_id",
            "relative_path",
            "sha256",
            "byte_size",
            "mime_type",
            "codec",
            "duration_ms",
            "complete",
            "deleted_at",
        ),
        "operation_receipts": (
            "workspace_id",
            "operation",
            "idempotency_key",
            "input_digest",
            "result_ref",
            "committed_at",
            "tombstone",
        ),
        "legacy_imports": (
            "source_fingerprint",
            "legacy_job_id",
            "production_job_id",
            "imported_at",
            "backup_reference",
            "warnings_json",
        ),
    }
    with sqlite3.connect(path) as db:
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]
        indexes = {
            (r[0], r[1])
            for r in db.execute(
                "SELECT name,tbl_name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
        }
        attempt_fks = {
            (r[2], r[3], r[4])
            for r in db.execute("PRAGMA foreign_key_list(generation_attempts)")
        }
        artifact_fks = {
            (r[2], r[3], r[4])
            for r in db.execute("PRAGMA foreign_key_list(production_artifacts)")
        }
    assert {"schema_migrations", *expected} <= tables
    for table, columns in expected.items():
        assert _columns(path, table) == columns
    assert indexes == {
        ("idx_voiceover_jobs_status_updated", "voiceover_jobs"),
        ("idx_voiceover_chunks_job", "voiceover_chunks"),
        ("idx_generation_attempts_chunk", "generation_attempts"),
        ("idx_production_artifacts_job", "production_artifacts"),
    }
    assert {
        ("voiceover_chunks", "job_id", "job_id"),
        ("voiceover_chunks", "chunk_id", "chunk_id"),
    } <= attempt_fks
    assert {
        ("generation_attempts", "attempt_id", "attempt_id"),
        ("generation_attempts", "job_id", "job_id"),
        ("generation_attempts", "chunk_id", "chunk_id"),
    } <= artifact_fks


@pytest.mark.asyncio
async def test_initialize_preserves_real_shaped_legacy_tables_and_rows(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import CREATE_JOBS_TABLE, CREATE_VOICES_TABLE, JobStore

    path = tmp_path / "voiceover_history.db"
    parts = [{"text": "historical", "voice_id": "voice-old"}]
    job = (
        "legacy-id",
        "completed",
        json.dumps(parts),
        "/old/output.mp3",
        None,
        "2024-01-01T00:00:00",
        "2024-01-01T00:01:00",
        1,
        1,
    )
    voice = (
        "voice-old",
        "Legacy Voice",
        "premade",
        json.dumps({"accent": "legacy"}),
        "description",
        "https://example.invalid/preview",
        json.dumps(["legacy_model"]),
        "2024-01-01T00:00:00",
    )
    with sqlite3.connect(path) as db:
        db.execute(CREATE_JOBS_TABLE)
        db.execute(CREATE_VOICES_TABLE)
        db.execute("INSERT INTO audio_jobs VALUES (?,?,?,?,?,?,?,?,?)", job)
        db.execute("INSERT INTO voices VALUES (?,?,?,?,?,?,?,?)", voice)
        db.commit()
    job_columns, voice_columns = _columns(path, "audio_jobs"), _columns(path, "voices")
    await JobStore(path).initialize()
    assert _columns(path, "audio_jobs") == job_columns
    assert _columns(path, "voices") == voice_columns
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM audio_jobs").fetchone() == job
        assert db.execute("SELECT * FROM voices").fetchone() == voice


@pytest.mark.asyncio
async def test_initialize_is_idempotent_and_concurrency_safe(tmp_path: Path) -> None:
    from elevenlabs_mcp.database import JobStore

    path = tmp_path / "voiceover_history.db"
    stores = [JobStore(path) for _ in range(8)]
    await asyncio.gather(*(store.initialize() for store in stores))
    code = (
        "import asyncio,sys; from elevenlabs_mcp.database import JobStore; "
        "asyncio.run(JobStore(sys.argv[1]).initialize())"
    )
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(4)
    ]
    for process in processes:
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, (stdout, stderr)
    await stores[0].initialize()
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT version,COUNT(*) FROM schema_migrations GROUP BY version"
        ).fetchall() == [(1, 1)]
        assert db.execute("PRAGMA user_version").fetchone() == (1,)


@pytest.mark.asyncio
async def test_newer_schema_version_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore, SchemaVersionError

    path = tmp_path / "future.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(SchemaVersionError) as caught:
        await JobStore(path).initialize()
    assert (caught.value.supported_version, caught.value.discovered_version) == (1, 99)
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            == []
        )
        assert db.execute("PRAGMA user_version").fetchone() == (99,)


@pytest.mark.asyncio
async def test_foreign_keys_reject_orphan_production_records(tmp_path: Path) -> None:
    from elevenlabs_mcp.database import JobStore

    store = JobStore(tmp_path / "voiceover_history.db")
    await store.initialize()
    async with store.connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(
                """INSERT INTO voiceover_chunks
            (chunk_id,job_id,chunk_index,generation_fingerprint,status,
             character_count,voice_ids_json,pause_after_ms,source_spans_json,request_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    "orphan",
                    "missing",
                    0,
                    "sha256:" + "0" * 64,
                    "pending",
                    1,
                    '["voice"]',
                    0,
                    "[]",
                    "{}",
                ),
            )
    with sqlite3.connect(store.db_path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        _insert_job(db, "job-a")
        _insert_job(db, "job-b")
        _insert_chunk(db, "chunk-a", "job-a")
        _insert_chunk(db, "chunk-b", "job-b")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO generation_attempts
            (attempt_id,job_id,chunk_id,dispatch_state,reserved_characters,started_at)
            VALUES (?,?,?,?,?,?)""",
                (
                    "attempt-cross",
                    "job-a",
                    "chunk-b",
                    "reserved",
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        db.execute(
            """INSERT INTO generation_attempts
        (attempt_id,job_id,chunk_id,dispatch_state,reserved_characters,started_at)
        VALUES (?,?,?,?,?,?)""",
            (
                "attempt-a",
                "job-a",
                "chunk-a",
                "reserved",
                1,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO production_artifacts
            (artifact_id,job_id,chunk_id,attempt_id,relative_path,sha256,byte_size,mime_type,complete)
            VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    "artifact-cross",
                    "job-b",
                    "chunk-b",
                    "attempt-a",
                    "file.mp3",
                    "sha256:" + "2" * 64,
                    1,
                    "audio/mpeg",
                    1,
                ),
            )
