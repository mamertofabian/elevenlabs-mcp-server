import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import AudioJob
from .contracts import JobCreateResult, VoiceoverPlan

CREATE_VOICES_TABLE = """
CREATE TABLE IF NOT EXISTS voices (
    voice_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    labels TEXT,  -- JSON string
    description TEXT,
    preview_url TEXT,
    high_quality_base_model_ids TEXT,  -- JSON string
    last_updated TEXT NOT NULL
)
"""

CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS audio_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    script_parts TEXT NOT NULL,  -- JSON string
    output_file TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total_parts INTEGER NOT NULL DEFAULT 1,
    completed_parts INTEGER NOT NULL DEFAULT 0
)
"""

_JOB_STORE_SCHEMA_VERSION = 2
_INITIALIZE_LOCKS: dict[str, asyncio.Lock] = {}
_JOB_STORE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS voiceover_jobs (
        job_id TEXT PRIMARY KEY,
        schema_version TEXT NOT NULL,
        plan_version TEXT NOT NULL,
        plan_hash TEXT NOT NULL,
        normalized_script_json TEXT NOT NULL,
        options_json TEXT NOT NULL,
        plan_json TEXT NOT NULL,
        status TEXT NOT NULL,
        reason TEXT,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        total_parts INTEGER NOT NULL CHECK (total_parts > 0),
        verified_parts INTEGER NOT NULL DEFAULT 0 CHECK (verified_parts >= 0),
        total_chunks INTEGER NOT NULL CHECK (total_chunks > 0),
        verified_chunks INTEGER NOT NULL DEFAULT 0 CHECK (verified_chunks >= 0),
        max_total_characters INTEGER NOT NULL CHECK (max_total_characters > 0),
        max_total_requests INTEGER NOT NULL CHECK (max_total_requests > 0),
        reserved_characters INTEGER NOT NULL DEFAULT 0 CHECK (reserved_characters >= 0),
        reserved_requests INTEGER NOT NULL DEFAULT 0 CHECK (reserved_requests >= 0),
        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
        source TEXT NOT NULL,
        deleted_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS voiceover_chunks (
        chunk_id TEXT NOT NULL,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        generation_fingerprint TEXT NOT NULL,
        status TEXT NOT NULL,
        character_count INTEGER NOT NULL CHECK (character_count > 0),
        voice_ids_json TEXT NOT NULL,
        pause_after_ms INTEGER NOT NULL DEFAULT 0 CHECK (pause_after_ms >= 0),
        source_spans_json TEXT NOT NULL,
        request_json TEXT NOT NULL,
        successful_artifact_id TEXT,
        latest_error TEXT,
        generation_metadata_json TEXT,
        PRIMARY KEY (job_id, chunk_id),
        UNIQUE (job_id, chunk_index),
        FOREIGN KEY (job_id, chunk_id, successful_artifact_id)
            REFERENCES production_artifacts(job_id, chunk_id, artifact_id)
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generation_attempts (
        attempt_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_id TEXT NOT NULL,
        dispatch_state TEXT NOT NULL,
        reserved_characters INTEGER NOT NULL CHECK (reserved_characters > 0),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        provider_request_id TEXT,
        sanitized_outcome TEXT,
        outcome_unknown INTEGER NOT NULL DEFAULT 0 CHECK (outcome_unknown IN (0, 1)),
        UNIQUE (job_id, chunk_id, attempt_id),
        FOREIGN KEY (job_id, chunk_id)
            REFERENCES voiceover_chunks(job_id, chunk_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS production_artifacts (
        artifact_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_id TEXT,
        attempt_id TEXT REFERENCES generation_attempts(attempt_id) ON DELETE RESTRICT,
        relative_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        mime_type TEXT NOT NULL,
        codec TEXT,
        duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
        complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
        deleted_at TEXT,
        UNIQUE (job_id, chunk_id, artifact_id),
        FOREIGN KEY (job_id, chunk_id)
            REFERENCES voiceover_chunks(job_id, chunk_id) ON DELETE RESTRICT,
        FOREIGN KEY (job_id, chunk_id, attempt_id)
            REFERENCES generation_attempts(job_id, chunk_id, attempt_id)
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS operation_receipts (
        workspace_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        input_digest TEXT NOT NULL,
        result_ref TEXT NOT NULL,
        committed_at TEXT NOT NULL,
        tombstone INTEGER NOT NULL DEFAULT 0 CHECK (tombstone IN (0, 1)),
        PRIMARY KEY (workspace_id, operation, idempotency_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS legacy_imports (
        source_fingerprint TEXT PRIMARY KEY,
        legacy_job_id TEXT NOT NULL,
        production_job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        imported_at TEXT NOT NULL,
        backup_reference TEXT,
        warnings_json TEXT NOT NULL,
        UNIQUE (legacy_job_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_voiceover_jobs_status_updated ON voiceover_jobs(status, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_voiceover_chunks_job ON voiceover_chunks(job_id, chunk_index)",
    "CREATE INDEX IF NOT EXISTS idx_generation_attempts_chunk ON generation_attempts(chunk_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_production_artifacts_job ON production_artifacts(job_id)",
)

_JOB_STORE_V1_TO_V2 = (
    "DROP INDEX IF EXISTS idx_voiceover_chunks_job",
    "DROP INDEX IF EXISTS idx_generation_attempts_chunk",
    "DROP INDEX IF EXISTS idx_production_artifacts_job",
    """
    CREATE TABLE voiceover_chunks_v2 (
        chunk_id TEXT NOT NULL,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        generation_fingerprint TEXT NOT NULL,
        status TEXT NOT NULL,
        character_count INTEGER NOT NULL CHECK (character_count > 0),
        voice_ids_json TEXT NOT NULL,
        pause_after_ms INTEGER NOT NULL DEFAULT 0 CHECK (pause_after_ms >= 0),
        source_spans_json TEXT NOT NULL,
        request_json TEXT NOT NULL,
        successful_artifact_id TEXT,
        latest_error TEXT,
        generation_metadata_json TEXT,
        PRIMARY KEY (job_id, chunk_id),
        UNIQUE (job_id, chunk_index),
        FOREIGN KEY (job_id, chunk_id, successful_artifact_id)
            REFERENCES production_artifacts_v2(job_id, chunk_id, artifact_id)
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE generation_attempts_v2 (
        attempt_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_id TEXT NOT NULL,
        dispatch_state TEXT NOT NULL,
        reserved_characters INTEGER NOT NULL CHECK (reserved_characters > 0),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        provider_request_id TEXT,
        sanitized_outcome TEXT,
        outcome_unknown INTEGER NOT NULL DEFAULT 0 CHECK (outcome_unknown IN (0, 1)),
        UNIQUE (job_id, chunk_id, attempt_id),
        FOREIGN KEY (job_id, chunk_id)
            REFERENCES voiceover_chunks_v2(job_id, chunk_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE production_artifacts_v2 (
        artifact_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_id TEXT,
        attempt_id TEXT REFERENCES generation_attempts_v2(attempt_id) ON DELETE RESTRICT,
        relative_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        mime_type TEXT NOT NULL,
        codec TEXT,
        duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
        complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
        deleted_at TEXT,
        UNIQUE (job_id, chunk_id, artifact_id),
        FOREIGN KEY (job_id, chunk_id)
            REFERENCES voiceover_chunks_v2(job_id, chunk_id) ON DELETE RESTRICT,
        FOREIGN KEY (job_id, chunk_id, attempt_id)
            REFERENCES generation_attempts_v2(job_id, chunk_id, attempt_id)
            ON DELETE RESTRICT
    )
    """,
    "INSERT INTO voiceover_chunks_v2 SELECT * FROM voiceover_chunks",
    "INSERT INTO generation_attempts_v2 SELECT * FROM generation_attempts",
    "INSERT INTO production_artifacts_v2 SELECT * FROM production_artifacts",
    "DROP TABLE production_artifacts",
    "DROP TABLE generation_attempts",
    "DROP TABLE voiceover_chunks",
    "ALTER TABLE voiceover_chunks_v2 RENAME TO voiceover_chunks",
    "ALTER TABLE generation_attempts_v2 RENAME TO generation_attempts",
    "ALTER TABLE production_artifacts_v2 RENAME TO production_artifacts",
    "CREATE INDEX idx_voiceover_chunks_job ON voiceover_chunks(job_id, chunk_index)",
    "CREATE INDEX idx_generation_attempts_chunk ON generation_attempts(chunk_id, started_at)",
    "CREATE INDEX idx_production_artifacts_job ON production_artifacts(job_id)",
)


class SchemaVersionError(RuntimeError):
    def __init__(self, supported_version: int, discovered_version: int) -> None:
        self.supported_version = supported_version
        self.discovered_version = discovered_version
        super().__init__(
            f"database schema version {discovered_version} is newer than supported "
            f"version {supported_version}"
        )


class IdempotencyConflictError(RuntimeError):
    def __init__(self, workspace_id: str, operation: str, idempotency_key: str) -> None:
        self.workspace_id = workspace_id
        self.operation = operation
        self.idempotency_key = idempotency_key
        super().__init__(f"idempotency key conflict for {workspace_id}/{operation}")


class BudgetExceededError(RuntimeError):
    def __init__(
        self,
        max_total_characters: int,
        required_characters: int,
        max_total_requests: int,
        required_requests: int,
    ) -> None:
        self.max_total_characters = max_total_characters
        self.required_characters = required_characters
        self.max_total_requests = max_total_requests
        self.required_requests = required_requests
        super().__init__("initial plan exceeds cumulative attempt ceiling")


class JobStore:
    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = os.fspath(db_path)
        lock_key = os.path.abspath(self.db_path)
        self._initialize_lock = _INITIALIZE_LOCKS.setdefault(lock_key, asyncio.Lock())

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        connection = await aiosqlite.connect(self.db_path, timeout=30.0)
        try:
            await connection.execute("PRAGMA busy_timeout = 30000")
            await connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            await connection.close()

    async def initialize(self) -> None:
        async with self._initialize_lock:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            async with self.connect() as connection:
                current = await _user_version(connection)
                if current > _JOB_STORE_SCHEMA_VERSION:
                    raise SchemaVersionError(_JOB_STORE_SCHEMA_VERSION, current)

                await connection.execute("PRAGMA journal_mode = WAL")
                if current == 1:
                    await connection.execute("PRAGMA foreign_keys = OFF")
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    current = await _user_version(connection)
                    if current > _JOB_STORE_SCHEMA_VERSION:
                        raise SchemaVersionError(_JOB_STORE_SCHEMA_VERSION, current)
                    if current == 0:
                        for statement in _JOB_STORE_SCHEMA:
                            await connection.execute(statement)
                        applied_at = datetime.now(UTC).isoformat()
                        await connection.executemany(
                            "INSERT INTO schema_migrations(version, applied_at) "
                            "VALUES (?, ?)",
                            ((1, applied_at), (2, applied_at)),
                        )
                    elif current == 1:
                        for statement in _JOB_STORE_V1_TO_V2:
                            await connection.execute(statement)
                        async with connection.execute(
                            "PRAGMA foreign_key_check"
                        ) as cursor:
                            violations = await cursor.fetchall()
                        if violations:
                            raise RuntimeError(
                                "database foreign-key check failed during migration"
                            )
                        await connection.execute(
                            "INSERT INTO schema_migrations(version, applied_at) "
                            "VALUES (?, ?)",
                            (2, datetime.now(UTC).isoformat()),
                        )
                    await connection.execute(
                        f"PRAGMA user_version = {_JOB_STORE_SCHEMA_VERSION}"
                    )
                    await connection.commit()
                except BaseException:
                    await connection.rollback()
                    raise
                finally:
                    await connection.execute("PRAGMA foreign_keys = ON")

    async def get_schema_version(self) -> int:
        async with self.connect() as connection:
            return await _user_version(connection)

    async def create_job(
        self,
        workspace_id: str,
        job_id: str,
        plan: VoiceoverPlan,
        idempotency_key: str,
        max_total_characters: int,
        max_total_requests: int,
        created_at: datetime,
    ) -> JobCreateResult:
        required_characters = sum(
            request.chunk.character_count for request in plan.requests
        )
        required_requests = len(plan.requests)
        digest_payload = {
            "max_total_characters": max_total_characters,
            "max_total_requests": max_total_requests,
            "plan": plan.model_dump(mode="json"),
        }
        digest_json = json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        input_digest = (
            "sha256:" + hashlib.sha256(digest_json.encode("utf-8")).hexdigest()
        )
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT input_digest, result_ref FROM operation_receipts
                    WHERE workspace_id = ? AND operation = 'submit'
                      AND idempotency_key = ?
                    """,
                    (workspace_id, idempotency_key),
                ) as cursor:
                    receipt = await cursor.fetchone()
                if receipt is not None:
                    if receipt[0] != input_digest:
                        raise IdempotencyConflictError(
                            workspace_id, "submit", idempotency_key
                        )
                    await connection.commit()
                    return JobCreateResult(
                        job_id=receipt[1],
                        revision=0,
                        created=False,
                        idempotent_replay=True,
                    )

                if (
                    required_characters > max_total_characters
                    or required_requests > max_total_requests
                ):
                    raise BudgetExceededError(
                        max_total_characters,
                        required_characters,
                        max_total_requests,
                        required_requests,
                    )
                if created_at.tzinfo is None or created_at.utcoffset() is None:
                    raise ValueError("created_at must be timezone-aware")
                timestamp = created_at.astimezone(UTC).isoformat()
                created_result = JobCreateResult(
                    job_id=job_id,
                    revision=0,
                    created=True,
                    idempotent_replay=False,
                )

                await connection.execute(
                    """
                    INSERT INTO voiceover_jobs (
                        job_id, schema_version, plan_version, plan_hash,
                        normalized_script_json, options_json, plan_json, status,
                        reason, revision, created_at, updated_at, total_parts,
                        verified_parts, total_chunks, verified_chunks,
                        max_total_characters, max_total_requests,
                        reserved_characters, reserved_requests, cancel_requested,
                        source, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        plan.schema_version,
                        plan.planner_version,
                        plan.plan_hash,
                        _json(plan.normalized_script.model_dump(mode="json")),
                        _json(plan.resolved_options.model_dump(mode="json")),
                        _json(plan.model_dump(mode="json")),
                        "queued",
                        None,
                        0,
                        timestamp,
                        timestamp,
                        plan.total_parts,
                        0,
                        len(plan.requests),
                        0,
                        max_total_characters,
                        max_total_requests,
                        0,
                        0,
                        0,
                        "native",
                        None,
                    ),
                )
                for request in plan.requests:
                    await connection.execute(
                        """
                        INSERT INTO voiceover_chunks (
                            chunk_id, job_id, chunk_index,
                            generation_fingerprint, status, character_count,
                            voice_ids_json, pause_after_ms, source_spans_json,
                            request_json, successful_artifact_id, latest_error,
                            generation_metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request.chunk_id,
                            job_id,
                            request.chunk.index,
                            request.generation_fingerprint,
                            "pending",
                            request.chunk.character_count,
                            _json(list(request.chunk.voice_ids)),
                            request.chunk.pause_after_ms,
                            _json(
                                [
                                    fragment.source_span.model_dump(mode="json")
                                    for fragment in request.chunk.fragments
                                ]
                            ),
                            _json(request.model_dump(mode="json")),
                            None,
                            None,
                            None,
                        ),
                    )
                await connection.execute(
                    """
                    INSERT INTO operation_receipts (
                        workspace_id, operation, idempotency_key, input_digest,
                        result_ref, committed_at, tombstone
                    ) VALUES (?, 'submit', ?, ?, ?, ?, 0)
                    """,
                    (workspace_id, idempotency_key, input_digest, job_id, timestamp),
                )
                await connection.commit()
                return created_result
            except BaseException:
                await connection.rollback()
                raise


async def _user_version(connection: aiosqlite.Connection) -> int:
    async with connection.execute("PRAGMA user_version") as cursor:
        row = await cursor.fetchone()
    return int(row[0]) if row is not None else 0


def _json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class Database:
    CACHE_DURATION_SECONDS = 24 * 60 * 60  # 24 hours

    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path: str = os.fspath(db_path)

    async def initialize(self) -> None:
        """Initialize database and create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            # Create tables one at a time
            await db.execute(CREATE_VOICES_TABLE)
            await db.execute(CREATE_JOBS_TABLE)
            await db.commit()

    async def insert_job(self, job: AudioJob) -> None:
        """Insert a new audio job into the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO audio_jobs 
                (id, status, script_parts, output_file, error, created_at, updated_at, total_parts, completed_parts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.status,
                    json.dumps(job.script_parts),
                    job.output_file,
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    job.total_parts,
                    job.completed_parts,
                ),
            )
            await db.commit()

    async def update_job(self, job: AudioJob) -> None:
        """Update an existing audio job in the database."""
        job.updated_at = datetime.utcnow()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE audio_jobs 
                SET status = ?, script_parts = ?, output_file = ?, error = ?, 
                    updated_at = ?, total_parts = ?, completed_parts = ?
                WHERE id = ?
                """,
                (
                    job.status,
                    json.dumps(job.script_parts),
                    job.output_file,
                    job.error,
                    job.updated_at.isoformat(),
                    job.total_parts,
                    job.completed_parts,
                    job.id,
                ),
            )
            await db.commit()

    async def get_job(self, job_id: str) -> Optional[AudioJob]:
        """Get a specific audio job by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM audio_jobs WHERE id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return AudioJob.from_dict(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "script_parts": json.loads(row["script_parts"]),
                        "output_file": row["output_file"],
                        "error": row["error"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "total_parts": row["total_parts"],
                        "completed_parts": row["completed_parts"],
                    }
                )

    async def get_all_jobs(self) -> List[AudioJob]:
        """Get all audio jobs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM audio_jobs ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    AudioJob.from_dict(
                        {
                            "id": row["id"],
                            "status": row["status"],
                            "script_parts": json.loads(row["script_parts"]),
                            "output_file": row["output_file"],
                            "error": row["error"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "total_parts": row["total_parts"],
                            "completed_parts": row["completed_parts"],
                        }
                    )
                    for row in rows
                ]

    async def delete_job(self, job_id: str) -> bool:
        """Delete an audio job by ID. Returns True if job was deleted."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM audio_jobs WHERE id = ?", (job_id,))
            deleted = cursor.rowcount > 0
            await db.commit()
            return deleted

    async def cleanup(self) -> None:
        """Delete the database file. Useful for testing."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def upsert_voices(self, voices: list[dict[str, object]]) -> None:
        """Insert or update voice data in the database."""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
            for voice in voices:
                await db.execute(
                    """
                    INSERT INTO voices 
                    (voice_id, name, category, labels, description, preview_url, 
                     high_quality_base_model_ids, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(voice_id) DO UPDATE SET
                        name = excluded.name,
                        category = excluded.category,
                        labels = excluded.labels,
                        description = excluded.description,
                        preview_url = excluded.preview_url,
                        high_quality_base_model_ids = excluded.high_quality_base_model_ids,
                        last_updated = excluded.last_updated
                    """,
                    (
                        voice["voice_id"],
                        voice["name"],
                        voice["category"],
                        json.dumps(voice["labels"]),
                        voice["description"],
                        voice["preview_url"],
                        json.dumps(voice["high_quality_base_model_ids"]),
                        now,
                    ),
                )
            await db.commit()

    async def get_voices(
        self, max_age_seconds: Optional[int] = None
    ) -> tuple[list[dict[str, object]], bool]:
        """
        Get all voices from the database.
        Returns tuple of (voices, needs_refresh) where needs_refresh indicates if cache is stale.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM voices ORDER BY name") as cursor:
                rows = await cursor.fetchall()

                voices = []
                needs_refresh = False

                if not rows:
                    needs_refresh = True
                else:
                    max_age = max_age_seconds or self.CACHE_DURATION_SECONDS
                    now = datetime.utcnow()

                    for row in rows:
                        last_updated = datetime.fromisoformat(row["last_updated"])
                        age = (now - last_updated).total_seconds()

                        if age > max_age:
                            needs_refresh = True

                        voices.append(
                            {
                                "voice_id": row["voice_id"],
                                "name": row["name"],
                                "category": row["category"],
                                "labels": json.loads(row["labels"]),
                                "description": row["description"],
                                "preview_url": row["preview_url"],
                                "high_quality_base_model_ids": json.loads(
                                    row["high_quality_base_model_ids"]
                                ),
                            }
                        )

                return voices, needs_refresh
