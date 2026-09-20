from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
NAIVE_NOW = NOW.replace(tzinfo=None)


def _plan(text: str = "hello"):
    from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
    from elevenlabs_mcp.planner import ScriptPlanner

    script = Script.model_validate(
        {
            "script_version": "1",
            "cast": {"actor": {"voice_id": "voice"}},
            "scenes": [
                {
                    "id": "scene",
                    "parts": [{"id": "part", "actor": "actor", "text": text}],
                }
            ],
        }
    )
    return ScriptPlanner().plan(
        script,
        VoiceoverOptions(engine="tts", model_id="eleven_multilingual_v2", seed=7),
        PlanningLimits(max_text_characters=3),
    )


def _create_schema_v1(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            CREATE TABLE voiceover_jobs(
              job_id TEXT PRIMARY KEY,schema_version TEXT NOT NULL,plan_version TEXT NOT NULL,
              plan_hash TEXT NOT NULL,normalized_script_json TEXT NOT NULL,options_json TEXT NOT NULL,
              plan_json TEXT NOT NULL,status TEXT NOT NULL,reason TEXT,revision INTEGER NOT NULL,
              created_at TEXT NOT NULL,updated_at TEXT NOT NULL,total_parts INTEGER NOT NULL,
              verified_parts INTEGER NOT NULL,total_chunks INTEGER NOT NULL,verified_chunks INTEGER NOT NULL,
              max_total_characters INTEGER NOT NULL,max_total_requests INTEGER NOT NULL,
              reserved_characters INTEGER NOT NULL,reserved_requests INTEGER NOT NULL,
              cancel_requested INTEGER NOT NULL,source TEXT NOT NULL,deleted_at TEXT);
            CREATE TABLE voiceover_chunks(
              chunk_id TEXT PRIMARY KEY,job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id),
              chunk_index INTEGER NOT NULL,generation_fingerprint TEXT NOT NULL,status TEXT NOT NULL,
              character_count INTEGER NOT NULL,voice_ids_json TEXT NOT NULL,pause_after_ms INTEGER NOT NULL,
              source_spans_json TEXT NOT NULL,request_json TEXT NOT NULL,successful_artifact_id TEXT,
              latest_error TEXT,generation_metadata_json TEXT,UNIQUE(job_id,chunk_index),UNIQUE(chunk_id,job_id));
            CREATE TABLE generation_attempts(
              attempt_id TEXT PRIMARY KEY,job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id),
              chunk_id TEXT NOT NULL REFERENCES voiceover_chunks(chunk_id),dispatch_state TEXT NOT NULL,
              reserved_characters INTEGER NOT NULL,started_at TEXT NOT NULL,ended_at TEXT,
              provider_request_id TEXT,sanitized_outcome TEXT,outcome_unknown INTEGER NOT NULL,
              UNIQUE(attempt_id,job_id,chunk_id),FOREIGN KEY(chunk_id,job_id) REFERENCES voiceover_chunks(chunk_id,job_id));
            CREATE TABLE production_artifacts(
              artifact_id TEXT PRIMARY KEY,job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id),
              chunk_id TEXT REFERENCES voiceover_chunks(chunk_id),attempt_id TEXT REFERENCES generation_attempts(attempt_id),
              relative_path TEXT NOT NULL,sha256 TEXT NOT NULL,byte_size INTEGER NOT NULL,mime_type TEXT NOT NULL,
              codec TEXT,duration_ms INTEGER,complete INTEGER NOT NULL,deleted_at TEXT,
              UNIQUE(artifact_id,job_id,chunk_id),FOREIGN KEY(chunk_id,job_id) REFERENCES voiceover_chunks(chunk_id,job_id),
              FOREIGN KEY(attempt_id,job_id,chunk_id) REFERENCES generation_attempts(attempt_id,job_id,chunk_id));
            CREATE TABLE operation_receipts(workspace_id TEXT,operation TEXT,idempotency_key TEXT,
              input_digest TEXT,result_ref TEXT,committed_at TEXT,tombstone INTEGER,
              PRIMARY KEY(workspace_id,operation,idempotency_key));
            CREATE TABLE legacy_imports(source_fingerprint TEXT PRIMARY KEY,legacy_job_id TEXT,
              production_job_id TEXT,imported_at TEXT,backup_reference TEXT,warnings_json TEXT);
            INSERT INTO schema_migrations VALUES(1,'2026-01-01T00:00:00+00:00');
            PRAGMA user_version=1;
            """
        )
        db.execute(
            """INSERT INTO voiceover_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "old-job",
                "1",
                "1",
                "sha256:" + "a" * 64,
                "{}",
                "{}",
                "{}",
                "completed",
                None,
                2,
                NOW.isoformat(),
                NOW.isoformat(),
                1,
                1,
                1,
                1,
                10,
                2,
                5,
                1,
                0,
                "native",
                None,
            ),
        )
        db.execute(
            "INSERT INTO voiceover_chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "old-chunk",
                "old-job",
                0,
                "sha256:" + "b" * 64,
                "succeeded",
                5,
                '["voice"]',
                0,
                '[{"part_id":"part","start":0,"end":5}]',
                "{}",
                None,
                None,
                "{}",
            ),
        )
        db.execute(
            "INSERT INTO generation_attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "old-attempt",
                "old-job",
                "old-chunk",
                "succeeded",
                5,
                NOW.isoformat(),
                NOW.isoformat(),
                "provider-id",
                "ok",
                0,
            ),
        )
        db.execute(
            "INSERT INTO production_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "old-artifact",
                "old-job",
                "old-chunk",
                "old-attempt",
                "old.mp3",
                "sha256:" + "c" * 64,
                5,
                "audio/mpeg",
                "mp3",
                100,
                1,
                None,
            ),
        )
        db.commit()


@pytest.mark.asyncio
async def test_create_job_atomically_persists_complete_plan_chunks_and_receipt(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.contracts import JobCreateResult
    from elevenlabs_mcp.database import JobStore

    store = JobStore(tmp_path / "db.sqlite")
    await store.initialize()
    plan = _plan()
    result = await store.create_job("workspace", "job-1", plan, "key-1", 100, 10, NOW)
    assert result == JobCreateResult(
        job_id="job-1", revision=0, created=True, idempotent_replay=False
    )
    with sqlite3.connect(store.db_path) as db:
        db.row_factory = sqlite3.Row
        job = db.execute("SELECT * FROM voiceover_jobs").fetchone()
        chunks = db.execute(
            "SELECT * FROM voiceover_chunks ORDER BY chunk_index"
        ).fetchall()
        receipt = db.execute("SELECT * FROM operation_receipts").fetchone()
    assert job["plan_hash"] == plan.plan_hash
    assert json.loads(
        job["normalized_script_json"]
    ) == plan.normalized_script.model_dump(mode="json")
    assert json.loads(job["options_json"]) == plan.resolved_options.model_dump(
        mode="json"
    )
    assert json.loads(job["plan_json"]) == plan.model_dump(mode="json")
    assert (
        job["status"],
        job["revision"],
        job["reserved_characters"],
        job["reserved_requests"],
    ) == ("queued", 0, 0, 0)
    assert len(chunks) == len(plan.requests)
    for row, request in zip(chunks, plan.requests, strict=True):
        assert row["chunk_id"] == request.chunk_id
        assert row["generation_fingerprint"] == request.generation_fingerprint
        assert json.loads(row["source_spans_json"]) == [
            f.source_span.model_dump(mode="json") for f in request.chunk.fragments
        ]
        assert json.loads(row["request_json"]) == request.model_dump(mode="json")
    assert (
        receipt["workspace_id"],
        receipt["operation"],
        receipt["idempotency_key"],
        receipt["result_ref"],
        receipt["tombstone"],
    ) == ("workspace", "submit", "key-1", "job-1", 0)
    expected_time = NOW.isoformat()
    assert (job["created_at"], job["updated_at"], receipt["committed_at"]) == (
        expected_time,
        expected_time,
        expected_time,
    )


@pytest.mark.asyncio
async def test_identical_replay_returns_original_result_and_conflicting_input_fails(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import IdempotencyConflictError, JobStore

    store = JobStore(tmp_path / "db.sqlite")
    await store.initialize()
    plan = _plan()
    first = await store.create_job("ws", "original", plan, "key", 100, 10, NOW)
    async with store.connect() as db:
        await db.execute(
            "UPDATE voiceover_jobs SET revision = 4 WHERE job_id = ?", ("original",)
        )
        await db.commit()
    replay = await store.create_job(
        "ws", "ignored-new-id", plan, "key", 100, 10, NAIVE_NOW
    )
    assert first.created is True
    assert (
        replay.job_id,
        replay.revision,
        replay.created,
        replay.idempotent_replay,
    ) == (
        "original",
        0,
        False,
        True,
    )
    with pytest.raises(IdempotencyConflictError) as caught:
        await store.create_job("ws", "other", _plan("changed"), "key", 100, 10, NOW)
    assert (
        caught.value.workspace_id,
        caught.value.operation,
        caught.value.idempotency_key,
    ) == ("ws", "submit", "key")
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM voiceover_jobs").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM operation_receipts").fetchone() == (1,)


@pytest.mark.asyncio
async def test_concurrent_identical_submit_creates_exactly_one_job(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    path = tmp_path / "db.sqlite"
    await JobStore(path).initialize()
    plan = _plan()
    stores = [JobStore(path) for _ in range(8)]
    results = await asyncio.gather(
        *(
            s.create_job("ws", f"job-{i}", plan, "key", 100, 10, NOW)
            for i, s in enumerate(stores)
        )
    )
    assert sum(r.created for r in results) == 1
    assert len({r.job_id for r in results}) == 1
    assert sum(r.idempotent_replay for r in results) == 7
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM voiceover_jobs").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM operation_receipts").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM voiceover_chunks").fetchone() == (
            len(plan.requests),
        )


@pytest.mark.asyncio
async def test_same_plan_can_create_independent_jobs_with_distinct_keys(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    store = JobStore(tmp_path / "db.sqlite")
    await store.initialize()
    plan = _plan()

    first = await store.create_job("ws", "job-1", plan, "key-1", 100, 10, NOW)
    second = await store.create_job("ws", "job-2", plan, "key-2", 100, 10, NOW)

    assert first.created is True
    assert second.created is True
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM voiceover_jobs").fetchone() == (2,)
        assert db.execute("SELECT COUNT(*) FROM operation_receipts").fetchone() == (2,)
        stored_chunks = db.execute(
            "SELECT job_id, chunk_id FROM voiceover_chunks ORDER BY job_id, chunk_index"
        ).fetchall()
    assert stored_chunks == [
        (job_id, request.chunk_id)
        for job_id in ("job-1", "job-2")
        for request in plan.requests
    ]


@pytest.mark.asyncio
async def test_schema_v1_migrates_to_job_scoped_chunks_without_data_loss(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    path = tmp_path / "schema-v1.sqlite"
    _create_schema_v1(path)
    store = JobStore(path)
    await store.initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (2,)
        assert db.execute("SELECT job_id, revision FROM voiceover_jobs").fetchall() == [
            ("old-job", 2)
        ]
        assert db.execute(
            "SELECT chunk_id, job_id FROM voiceover_chunks"
        ).fetchall() == [("old-chunk", "old-job")]
        assert db.execute("SELECT attempt_id FROM generation_attempts").fetchall() == [
            ("old-attempt",)
        ]
        assert db.execute(
            "SELECT artifact_id FROM production_artifacts"
        ).fetchall() == [("old-artifact",)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        pk = {
            row[1]: row[5]
            for row in db.execute("PRAGMA table_info(voiceover_chunks)")
            if row[5]
        }
    assert pk == {"job_id": 1, "chunk_id": 2}

    plan = _plan()
    await store.create_job("ws", "job-1", plan, "key-1", 100, 10, NOW)
    await store.create_job("ws", "job-2", plan, "key-2", 100, 10, NOW)


@pytest.mark.asyncio
async def test_schema_v1_integrity_failure_rolls_back_entire_migration(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    path = tmp_path / "invalid-v1.sqlite"
    _create_schema_v1(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO generation_attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "invalid-attempt",
                "missing-job",
                "old-chunk",
                "succeeded",
                5,
                NOW.isoformat(),
                NOW.isoformat(),
                None,
                "invalid ownership",
                0,
            ),
        )
        db.commit()

    store = JobStore(path)
    with pytest.raises(RuntimeError, match="foreign-key check failed"):
        await store.initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (1,)
        assert db.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]
        assert db.execute("SELECT COUNT(*) FROM generation_attempts").fetchone() == (2,)
        pk = {
            row[1]: row[5]
            for row in db.execute("PRAGMA table_info(voiceover_chunks)")
            if row[5]
        }
    assert pk == {"chunk_id": 1}
    async with (
        store.connect() as connection,
        connection.execute("PRAGMA foreign_keys") as cursor,
    ):
        assert await cursor.fetchone() == (1,)


@pytest.mark.asyncio
async def test_invalid_new_job_id_persists_nothing_but_replay_ignores_it(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    store = JobStore(tmp_path / "db.sqlite")
    await store.initialize()
    plan = _plan()
    for invalid_id in ("", "x" * 129):
        with pytest.raises(ValidationError):
            await store.create_job(
                "ws", invalid_id, plan, invalid_id or "empty", 100, 10, NOW
            )
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM voiceover_jobs").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM operation_receipts").fetchone() == (0,)

    await store.create_job("ws", "valid", plan, "committed", 100, 10, NOW)
    replay = await store.create_job("ws", "", plan, "committed", 100, 10, NAIVE_NOW)
    assert (replay.job_id, replay.revision, replay.idempotent_replay) == (
        "valid",
        0,
        True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("characters", "requests"), [(4, 10), (100, 1)])
async def test_insufficient_initial_ceiling_leaves_database_empty(
    tmp_path: Path, characters: int, requests: int
) -> None:
    from elevenlabs_mcp.database import BudgetExceededError, JobStore

    store = JobStore(tmp_path / f"{characters}-{requests}.sqlite")
    await store.initialize()
    plan = _plan()
    with pytest.raises(BudgetExceededError) as caught:
        await store.create_job("ws", "job", plan, "key", characters, requests, NOW)
    assert caught.value.max_total_characters == characters
    assert caught.value.required_characters == 5
    assert caught.value.max_total_requests == requests
    assert caught.value.required_requests == 2
    with sqlite3.connect(store.db_path) as db:
        for table in ("voiceover_jobs", "voiceover_chunks", "operation_receipts"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


@pytest.mark.asyncio
async def test_chunk_insert_failure_rolls_back_job_chunks_and_receipt(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.database import JobStore

    store = JobStore(tmp_path / "db.sqlite")
    await store.initialize()
    async with store.connect() as db:
        await db.execute(
            "CREATE TRIGGER reject_chunks BEFORE INSERT ON voiceover_chunks BEGIN SELECT RAISE(ABORT,'fault'); END"
        )
        await db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        await store.create_job("ws", "job", _plan(), "key", 100, 10, NOW)
    with sqlite3.connect(store.db_path) as db:
        for table in ("voiceover_jobs", "voiceover_chunks", "operation_receipts"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
