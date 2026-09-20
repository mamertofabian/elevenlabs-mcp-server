import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import AudioJob

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

_JOB_STORE_SCHEMA_VERSION = 1
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
        chunk_id TEXT PRIMARY KEY,
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
        UNIQUE (job_id, chunk_index),
        UNIQUE (chunk_id, job_id),
        FOREIGN KEY (successful_artifact_id, job_id, chunk_id)
            REFERENCES production_artifacts(artifact_id, job_id, chunk_id)
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generation_attempts (
        attempt_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_id TEXT NOT NULL REFERENCES voiceover_chunks(chunk_id) ON DELETE RESTRICT,
        dispatch_state TEXT NOT NULL,
        reserved_characters INTEGER NOT NULL CHECK (reserved_characters > 0),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        provider_request_id TEXT,
        sanitized_outcome TEXT,
        outcome_unknown INTEGER NOT NULL DEFAULT 0 CHECK (outcome_unknown IN (0, 1)),
        UNIQUE (attempt_id, job_id, chunk_id),
        FOREIGN KEY (chunk_id, job_id)
            REFERENCES voiceover_chunks(chunk_id, job_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS production_artifacts (
        artifact_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES voiceover_jobs(job_id) ON DELETE RESTRICT,
        chunk_id TEXT REFERENCES voiceover_chunks(chunk_id) ON DELETE RESTRICT,
        attempt_id TEXT REFERENCES generation_attempts(attempt_id) ON DELETE RESTRICT,
        relative_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        mime_type TEXT NOT NULL,
        codec TEXT,
        duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
        complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
        deleted_at TEXT,
        UNIQUE (artifact_id, job_id, chunk_id),
        FOREIGN KEY (chunk_id, job_id)
            REFERENCES voiceover_chunks(chunk_id, job_id) ON DELETE RESTRICT,
        FOREIGN KEY (attempt_id, job_id, chunk_id)
            REFERENCES generation_attempts(attempt_id, job_id, chunk_id)
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


class SchemaVersionError(RuntimeError):
    def __init__(self, supported_version: int, discovered_version: int) -> None:
        self.supported_version = supported_version
        self.discovered_version = discovered_version
        super().__init__(
            f"database schema version {discovered_version} is newer than supported "
            f"version {supported_version}"
        )


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
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    current = await _user_version(connection)
                    if current > _JOB_STORE_SCHEMA_VERSION:
                        raise SchemaVersionError(_JOB_STORE_SCHEMA_VERSION, current)
                    if current < _JOB_STORE_SCHEMA_VERSION:
                        for statement in _JOB_STORE_SCHEMA:
                            await connection.execute(statement)
                        await connection.execute(
                            "INSERT INTO schema_migrations(version, applied_at) "
                            "VALUES (?, ?)",
                            (
                                _JOB_STORE_SCHEMA_VERSION,
                                datetime.now(UTC).isoformat(),
                            ),
                        )
                        await connection.execute(
                            f"PRAGMA user_version = {_JOB_STORE_SCHEMA_VERSION}"
                        )
                    await connection.commit()
                except BaseException:
                    await connection.rollback()
                    raise

    async def get_schema_version(self) -> int:
        async with self.connect() as connection:
            return await _user_version(connection)


async def _user_version(connection: aiosqlite.Connection) -> int:
    async with connection.execute("PRAGMA user_version") as cursor:
        row = await cursor.fetchone()
    return int(row[0]) if row is not None else 0


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
