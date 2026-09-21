import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Literal, Optional

import aiosqlite

from .artifacts import CompletionRecord
from .audio import AudioVerificationResult
from .contracts import (
    ArtifactRecordResult,
    AttemptDispatch,
    AttemptFailureResult,
    AttemptReservation,
    CancellationResult,
    JobCreateResult,
    RestartReconciliationResult,
    ResumeResult,
    VoiceoverPlan,
)
from .models import AudioJob
from .workspace import WorkspaceLock, WorkspaceOwnershipError

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


class JobNotFoundError(RuntimeError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"job not found: {job_id}")


class ChunkNotFoundError(RuntimeError):
    def __init__(self, job_id: str, chunk_id: str) -> None:
        self.job_id = job_id
        self.chunk_id = chunk_id
        super().__init__(f"chunk not found: {job_id}/{chunk_id}")


class AttemptConflictError(RuntimeError):
    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        super().__init__(f"attempt ID already belongs to another request: {attempt_id}")


class ReservationStateError(RuntimeError):
    def __init__(
        self, job_id: str, chunk_id: str, job_status: str, chunk_status: str
    ) -> None:
        self.job_id = job_id
        self.chunk_id = chunk_id
        self.job_status = job_status
        self.chunk_status = chunk_status
        super().__init__(
            f"cannot reserve {job_id}/{chunk_id} from {job_status}/{chunk_status}"
        )


class AttemptNotFoundError(RuntimeError):
    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        super().__init__(f"attempt not found: {attempt_id}")


class AttemptStateError(RuntimeError):
    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        super().__init__(f"attempt cannot be dispatched: {attempt_id}")


class AttemptOutcomeConflictError(RuntimeError):
    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        super().__init__(f"attempt outcome already recorded differently: {attempt_id}")


class RevisionConflictError(RuntimeError):
    def __init__(self, job_id: str, expected_revision: int, actual_revision: int) -> None:
        self.job_id = job_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(f"job revision conflict: {job_id}")


class JobBusyError(RuntimeError):
    """Job activity must finish or be reconciled before this mutation."""


class UncertainAttemptError(RuntimeError):
    """Resuming uncertain work requires explicit duplicate-charge acknowledgment."""


class ArtifactVerificationRequiredError(RuntimeError):
    """Successful artifacts require trusted verification before resume."""


class ArtifactConflictError(RuntimeError):
    """Artifact metadata conflicts with ownership, an immutable plan, or prior evidence."""


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

    async def reserve_attempt(
        self,
        job_id: str,
        chunk_id: str,
        attempt_id: str,
        started_at: datetime,
    ) -> AttemptReservation:
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT job_id, chunk_id, reserved_characters
                    FROM generation_attempts WHERE attempt_id = ?
                    """,
                    (attempt_id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    if (existing[0], existing[1]) != (job_id, chunk_id):
                        raise AttemptConflictError(attempt_id)
                    await connection.commit()
                    return AttemptReservation(
                        attempt_id=attempt_id,
                        job_id=job_id,
                        chunk_id=chunk_id,
                        reserved_characters=existing[2],
                        reserved_requests=1,
                        replayed=True,
                    )

                async with connection.execute(
                    """
                    SELECT max_total_characters, max_total_requests,
                           reserved_characters, reserved_requests, status, cancel_requested
                    FROM voiceover_jobs
                    WHERE job_id = ? AND deleted_at IS NULL
                    """,
                    (job_id,),
                ) as cursor:
                    job = await cursor.fetchone()
                if job is None:
                    raise JobNotFoundError(job_id)

                async with connection.execute(
                    """
                    SELECT character_count, status FROM voiceover_chunks
                    WHERE job_id = ? AND chunk_id = ?
                    """,
                    (job_id, chunk_id),
                ) as cursor:
                    chunk = await cursor.fetchone()
                if chunk is None:
                    raise ChunkNotFoundError(job_id, chunk_id)
                if job[4] not in {"queued", "running"} or job[5] or chunk[1] != "pending":
                    raise ReservationStateError(job_id, chunk_id, job[4], chunk[1])

                required_characters = job[2] + chunk[0]
                required_requests = job[3] + 1
                if required_characters > job[0] or required_requests > job[1]:
                    raise BudgetExceededError(
                        job[0], required_characters, job[1], required_requests
                    )
                if started_at.tzinfo is None or started_at.utcoffset() is None:
                    raise ValueError("started_at must be timezone-aware")
                result = AttemptReservation(
                    attempt_id=attempt_id,
                    job_id=job_id,
                    chunk_id=chunk_id,
                    reserved_characters=chunk[0],
                    reserved_requests=1,
                    replayed=False,
                )
                await connection.execute(
                    """
                    UPDATE voiceover_jobs
                    SET reserved_characters = ?, reserved_requests = ?,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        required_characters,
                        required_requests,
                        started_at.astimezone(UTC).isoformat(),
                        job_id,
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO generation_attempts (
                        attempt_id, job_id, chunk_id, dispatch_state,
                        reserved_characters, started_at, outcome_unknown
                    ) VALUES (?, ?, ?, 'reserved', ?, ?, 0)
                    """,
                    (
                        attempt_id,
                        job_id,
                        chunk_id,
                        chunk[0],
                        started_at.astimezone(UTC).isoformat(),
                    ),
                )
                await connection.commit()
                return result
            except BaseException:
                await connection.rollback()
                raise

    async def mark_attempt_dispatched(
        self, attempt_id: str, dispatched_at: datetime
    ) -> AttemptDispatch:
        """Commit dispatch intent before HTTP; duplicate calls fail closed.

        A caller may dispatch only after a successful return, exactly once. A
        crash after commit is conservatively uncertain, not permission to replay.
        Reservations remain charged against the local exposure ceiling.
        """
        if dispatched_at.tzinfo is None or dispatched_at.utcoffset() is None:
            raise ValueError("dispatched_at must be timezone-aware")
        dispatched_at = dispatched_at.astimezone(UTC)
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT a.job_id, a.chunk_id, a.dispatch_state, a.started_at,
                           j.status, j.cancel_requested, j.deleted_at, c.status
                    FROM generation_attempts a
                    JOIN voiceover_jobs j ON j.job_id = a.job_id
                    JOIN voiceover_chunks c
                      ON c.job_id = a.job_id AND c.chunk_id = a.chunk_id
                    WHERE a.attempt_id = ?
                    """,
                    (attempt_id,),
                ) as cursor:
                    attempt = await cursor.fetchone()
                if attempt is None:
                    raise AttemptNotFoundError(attempt_id)
                if (
                    attempt[2] != "reserved"
                    or attempt[4] not in {"queued", "running"}
                    or attempt[5]
                    or attempt[6] is not None
                    or attempt[7] != "pending"
                ):
                    raise AttemptStateError(attempt_id)
                if dispatched_at < datetime.fromisoformat(attempt[3]):
                    raise ValueError("dispatch cannot precede reservation")
                result = AttemptDispatch(
                    attempt_id=attempt_id, job_id=attempt[0], chunk_id=attempt[1]
                )
                await connection.execute(
                    "UPDATE generation_attempts SET dispatch_state = 'dispatched' "
                    "WHERE attempt_id = ?",
                    (attempt_id,),
                )
                await connection.execute(
                    "UPDATE voiceover_chunks SET status = 'generating' "
                    "WHERE job_id = ? AND chunk_id = ?",
                    (attempt[0], attempt[1]),
                )
                await connection.execute(
                    "UPDATE voiceover_jobs SET status = 'running', updated_at = ? "
                    "WHERE job_id = ?",
                    (dispatched_at.isoformat(), attempt[0]),
                )
                await connection.commit()
                return result
            except BaseException:
                await connection.rollback()
                raise

    async def record_attempt_failure(
        self,
        attempt_id: str,
        outcome: Literal["failed", "unknown"],
        ended_at: datetime,
        provider_request_id: str | None = None,
    ) -> AttemptFailureResult:
        """Finalize a dispatched failure without refunding exposure or retrying.

        Only fixed outcome codes enter diagnostics, never raw provider errors.
        Identical retries preserve the first terminal timestamp and all job state.
        Unknown outcomes pause the job until an explicit recovery decision.
        """
        if outcome not in {"failed", "unknown"}:
            raise ValueError("outcome must be failed or unknown")
        if ended_at.tzinfo is None or ended_at.utcoffset() is None:
            raise ValueError("ended_at must be timezone-aware")
        if provider_request_id is not None and (
            not isinstance(provider_request_id, str)
            or not 1 <= len(provider_request_id) <= 128
        ):
            raise ValueError("provider_request_id must contain 1 to 128 characters")
        ended_at = ended_at.astimezone(UTC)
        reason = "UPSTREAM_OUTCOME_UNKNOWN" if outcome == "unknown" else "GENERATION_FAILED"
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT a.job_id, a.chunk_id, a.dispatch_state, a.started_at,
                           a.provider_request_id, j.status, j.deleted_at, c.status
                    FROM generation_attempts a
                    JOIN voiceover_jobs j ON j.job_id = a.job_id
                    JOIN voiceover_chunks c
                      ON c.job_id = a.job_id AND c.chunk_id = a.chunk_id
                    WHERE a.attempt_id = ?
                    """,
                    (attempt_id,),
                ) as cursor:
                    attempt = await cursor.fetchone()
                if attempt is None:
                    raise AttemptNotFoundError(attempt_id)
                replayed = attempt[2] in {"failed", "unknown"}
                result = AttemptFailureResult(
                    attempt_id=attempt_id, job_id=attempt[0], chunk_id=attempt[1],
                    outcome=outcome, replayed=replayed,
                )
                if replayed:
                    if attempt[2] != outcome or attempt[4] != provider_request_id:
                        raise AttemptOutcomeConflictError(attempt_id)
                    await connection.commit()
                    return result
                if (
                    attempt[2] != "dispatched"
                    or attempt[5] not in {"running", "paused", "failed"}
                    or attempt[6] is not None
                    or attempt[7] != "generating"
                ):
                    raise AttemptStateError(attempt_id)
                if ended_at < datetime.fromisoformat(attempt[3]):
                    raise ValueError("attempt cannot end before reservation")
                await connection.execute(
                    """
                    UPDATE generation_attempts
                    SET dispatch_state = ?, ended_at = ?, provider_request_id = ?,
                        sanitized_outcome = ?, outcome_unknown = ?
                    WHERE attempt_id = ?
                    """,
                    (outcome, ended_at.isoformat(), provider_request_id, reason,
                     int(outcome == "unknown"), attempt_id),
                )
                await connection.execute(
                    "UPDATE voiceover_chunks SET status = ?, latest_error = ? "
                    "WHERE job_id = ? AND chunk_id = ?",
                    (outcome, reason, attempt[0], attempt[1]),
                )
                async with connection.execute(
                    "SELECT 1 FROM voiceover_chunks WHERE job_id = ? "
                    "AND status = 'unknown' LIMIT 1", (attempt[0],),
                ) as cursor:
                    has_unknown = await cursor.fetchone() is not None
                await connection.execute(
                    "UPDATE voiceover_jobs SET status = ?, reason = ?, updated_at = ? "
                    "WHERE job_id = ?",
                    ("paused" if has_unknown else "failed",
                     "UPSTREAM_OUTCOME_UNKNOWN" if has_unknown else reason,
                     ended_at.isoformat(), attempt[0]),
                )
                await connection.commit()
                return result
            except BaseException:
                await connection.rollback()
                raise

    async def request_cancel(
        self,
        workspace_id: str,
        job_id: str,
        expected_revision: int,
        idempotency_key: str,
        requested_at: datetime,
    ) -> CancellationResult:
        """Persist cancellation intent and its original acknowledgment atomically.

        Receipt lookup precedes revision checks. Workers must reconcile active
        attempts before marking work cancelled; this method only blocks scheduling.
        """
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        for value in (workspace_id, job_id, idempotency_key):
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError("workspace, job and idempotency IDs require 1 to 128 characters")
        if requested_at.tzinfo is None or requested_at.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware")
        timestamp = requested_at.astimezone(UTC).isoformat()
        digest = "sha256:" + hashlib.sha256(_json({
            "job_id": job_id, "expected_revision": expected_revision,
        }).encode("utf-8")).hexdigest()
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT input_digest, result_ref FROM operation_receipts "
                    "WHERE workspace_id = ? AND operation = 'cancel' AND idempotency_key = ?",
                    (workspace_id, idempotency_key),
                ) as cursor:
                    receipt = await cursor.fetchone()
                if receipt is not None:
                    if receipt[0] != digest:
                        raise IdempotencyConflictError(workspace_id, "cancel", idempotency_key)
                    # Cancel receipts embed the original acknowledgment, not mutable job state.
                    result = CancellationResult.model_validate_json(receipt[1])
                    await connection.commit()
                    return result.model_copy(update={"replayed": True})
                async with connection.execute(
                    "SELECT revision, status, reason, cancel_requested FROM voiceover_jobs "
                    "WHERE job_id = ? AND deleted_at IS NULL", (job_id,),
                ) as cursor:
                    job = await cursor.fetchone()
                if job is None:
                    raise JobNotFoundError(job_id)
                if job[0] != expected_revision:
                    raise RevisionConflictError(job_id, expected_revision, job[0])
                change = job[1] not in {"completed", "cancelled"} and not job[3]
                result = CancellationResult(
                    job_id=job_id, revision=job[0] + int(change), status=job[1],
                    reason=job[2], cancel_requested=bool(job[3]) or change,
                    replayed=False,
                )
                if change:
                    await connection.execute(
                        "UPDATE voiceover_jobs SET cancel_requested = 1, revision = ?, updated_at = ? "
                        "WHERE job_id = ?",
                        (result.revision, timestamp, job_id),
                    )
                await connection.execute(
                    """
                    INSERT INTO operation_receipts (
                        workspace_id, operation, idempotency_key, input_digest,
                        result_ref, committed_at, tombstone
                    ) VALUES (?, 'cancel', ?, ?, ?, ?, 0)
                    """,
                    (workspace_id, idempotency_key, digest,
                     result.model_dump_json(), timestamp),
                )
                await connection.commit()
                return result
            except BaseException:
                await connection.rollback()
                raise

    async def request_resume(
        self,
        workspace_id: str,
        job_id: str,
        expected_revision: int,
        idempotency_key: str,
        max_total_characters: int,
        max_total_requests: int,
        requested_at: datetime,
        retry_uncertain: bool = False,
    ) -> ResumeResult:
        """Authorize fresh attempts without resetting historical exposure.

        This storage primitive refuses successful chunks until artifact verification
        is implemented. It never treats an unverified file as reusable audio.
        """
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        for value in (max_total_characters, max_total_requests):
            if type(value) is not int or value < 1:
                raise ValueError("attempt ceilings must be positive integers")
        if type(retry_uncertain) is not bool:
            raise ValueError("retry_uncertain must be a boolean")
        for value in (workspace_id, job_id, idempotency_key):
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError("workspace, job and idempotency IDs require 1 to 128 characters")
        if requested_at.tzinfo is None or requested_at.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware")
        timestamp = requested_at.astimezone(UTC).isoformat()
        digest = "sha256:" + hashlib.sha256(_json({
            "job_id": job_id, "expected_revision": expected_revision,
            "max_total_characters": max_total_characters,
            "max_total_requests": max_total_requests,
            "retry_uncertain": retry_uncertain,
        }).encode("utf-8")).hexdigest()
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT input_digest, result_ref FROM operation_receipts "
                    "WHERE workspace_id = ? AND operation = 'resume' AND idempotency_key = ?",
                    (workspace_id, idempotency_key),
                ) as cursor:
                    receipt = await cursor.fetchone()
                if receipt is not None:
                    if receipt[0] != digest:
                        raise IdempotencyConflictError(workspace_id, "resume", idempotency_key)
                    result = ResumeResult.model_validate_json(receipt[1])
                    await connection.commit()
                    return result.model_copy(update={"replayed": True})
                async with connection.execute(
                    "SELECT revision, status, reserved_characters, reserved_requests "
                    "FROM voiceover_jobs WHERE job_id = ? AND deleted_at IS NULL",
                    (job_id,),
                ) as cursor:
                    job = await cursor.fetchone()
                if job is None:
                    raise JobNotFoundError(job_id)
                if job[0] != expected_revision:
                    raise RevisionConflictError(job_id, expected_revision, job[0])
                warnings: tuple[str, ...] = ()
                completed = job[1] == "completed"
                if not completed:
                    if job[1] not in {"paused", "failed", "cancelled"}:
                        raise JobBusyError(job_id)
                    async with connection.execute(
                        "SELECT status FROM voiceover_chunks WHERE job_id = ?", (job_id,),
                    ) as cursor:
                        states = {row[0] for row in await cursor.fetchall()}
                    async with connection.execute(
                        "SELECT 1 FROM generation_attempts WHERE job_id = ? "
                        "AND dispatch_state = 'dispatched' LIMIT 1", (job_id,),
                    ) as cursor:
                        inflight = await cursor.fetchone() is not None
                    if inflight or "generating" in states:
                        raise JobBusyError(job_id)
                    if "succeeded" in states:
                        raise ArtifactVerificationRequiredError(job_id)
                    if not states or not states <= {"pending", "failed", "unknown", "cancelled"}:
                        raise JobBusyError(job_id)
                    if "unknown" in states:
                        if not retry_uncertain:
                            raise UncertainAttemptError(job_id)
                        warnings = ("Retrying uncertain synthesis may incur duplicate charges.",)
                    if max_total_characters < job[2] or max_total_requests < job[3]:
                        raise BudgetExceededError(max_total_characters, job[2], max_total_requests, job[3])
                    # Retire undispatched reservations so stale callers cannot use
                    # an old attempt after this explicit new authorization.
                    await connection.execute(
                        "UPDATE generation_attempts SET dispatch_state = 'cancelled', "
                        "ended_at = ?, sanitized_outcome = 'RESUME_ABANDONED_RESERVATION' "
                        "WHERE job_id = ? AND dispatch_state = 'reserved'",
                        (timestamp, job_id),
                    )
                    await connection.execute(
                        "UPDATE voiceover_chunks SET status = 'pending' WHERE job_id = ?",
                        (job_id,),
                    )
                    await connection.execute(
                        "UPDATE voiceover_jobs SET status = 'queued', reason = NULL, "
                        "cancel_requested = 0, revision = ?, updated_at = ?, "
                        "max_total_characters = ?, max_total_requests = ? WHERE job_id = ?",
                        (job[0] + 1, timestamp, max_total_characters, max_total_requests, job_id),
                    )
                result = ResumeResult(
                    job_id=job_id, revision=job[0] + int(not completed),
                    status="completed" if completed else "queued",
                    warnings=warnings, replayed=False,
                )
                await connection.execute(
                    "INSERT INTO operation_receipts (workspace_id, operation, idempotency_key, "
                    "input_digest, result_ref, committed_at, tombstone) "
                    "VALUES (?, 'resume', ?, ?, ?, ?, 0)",
                    (workspace_id, idempotency_key, digest, result.model_dump_json(), timestamp),
                )
                await connection.commit()
                return result
            except BaseException:
                await connection.rollback()
                raise

    async def record_verified_artifact(
        self,
        artifact_id: str,
        verification: AudioVerificationResult,
        completed_at: datetime,
    ) -> ArtifactRecordResult:
        """Record trusted verifier evidence; never perform file or media I/O in SQL.

        This is an internal persistence boundary, not a client-supplied attestation.
        The source must be checked again before reuse: recorded evidence describes
        the verifier's snapshot, not a promise that the path remains unchanged.
        """
        verification = AudioVerificationResult.model_validate(verification.model_dump())
        integrity = verification.integrity
        identity = integrity.identity
        CompletionRecord(
            schema_version="1", identity=identity, sha256=integrity.sha256,
            byte_size=integrity.byte_size,
        )
        relative_path = (
            f"jobs/{identity.job_id}/chunks/{identity.chunk_id}/{identity.attempt_id}.mp3"
        )
        if integrity.relative_path != relative_path:
            raise ArtifactConflictError("artifact path does not match attempt ownership")
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        completed_at = completed_at.astimezone(UTC)
        result = ArtifactRecordResult(
            artifact_id=artifact_id, job_id=identity.job_id, chunk_id=identity.chunk_id,
            attempt_id=identity.attempt_id, replayed=False,
        )
        expected = (
            identity.job_id, identity.chunk_id, identity.attempt_id, relative_path,
            integrity.sha256, integrity.byte_size, "audio/mpeg", "mp3",
            verification.duration_ms, 1, None,
        )
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT a.job_id, a.chunk_id, a.dispatch_state, a.started_at,
                           c.generation_fingerprint, c.status, j.status, j.deleted_at
                    FROM generation_attempts a
                    JOIN voiceover_chunks c ON c.job_id=a.job_id AND c.chunk_id=a.chunk_id
                    JOIN voiceover_jobs j ON j.job_id=a.job_id
                    WHERE a.attempt_id = ?
                    """, (identity.attempt_id,),
                ) as cursor:
                    attempt = await cursor.fetchone()
                if attempt is None:
                    raise AttemptNotFoundError(identity.attempt_id)
                if (attempt[0], attempt[1], attempt[4]) != (
                    identity.job_id, identity.chunk_id, identity.generation_fingerprint,
                ):
                    raise ArtifactConflictError("artifact does not match owned plan")
                async with connection.execute(
                    "SELECT job_id,chunk_id,attempt_id,relative_path,sha256,byte_size,"
                    "mime_type,codec,duration_ms,complete,deleted_at "
                    "FROM production_artifacts WHERE artifact_id = ?", (artifact_id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    if tuple(existing) != expected:
                        raise ArtifactConflictError("artifact ID already has different evidence")
                    await connection.commit()
                    return result.model_copy(update={"replayed": True})
                if (
                    attempt[2] != "dispatched" or attempt[5] != "generating"
                    or attempt[6] not in {"running", "paused", "failed"}
                    or attempt[7] is not None
                ):
                    raise AttemptStateError(identity.attempt_id)
                if completed_at < datetime.fromisoformat(attempt[3]):
                    raise ValueError("completion cannot precede reservation")
                await connection.execute(
                    "INSERT INTO production_artifacts (artifact_id,job_id,chunk_id,attempt_id,"
                    "relative_path,sha256,byte_size,mime_type,codec,duration_ms,complete,deleted_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, *expected),
                )
                await connection.execute(
                    "UPDATE generation_attempts SET dispatch_state='succeeded', ended_at=?, "
                    "sanitized_outcome='AUDIO_VERIFIED', outcome_unknown=0 WHERE attempt_id=?",
                    (completed_at.isoformat(), identity.attempt_id),
                )
                await connection.execute(
                    "UPDATE voiceover_chunks SET status='succeeded', successful_artifact_id=?, "
                    "latest_error=NULL WHERE job_id=? AND chunk_id=?",
                    (artifact_id, identity.job_id, identity.chunk_id),
                )
                async with connection.execute(
                    "SELECT status,source_spans_json FROM voiceover_chunks WHERE job_id=?",
                    (identity.job_id,),
                ) as cursor:
                    chunks = await cursor.fetchall()
                all_parts: set[str] = set()
                incomplete_parts: set[str] = set()
                verified_chunks = 0
                for status, spans_json in chunks:
                    parts = {span["part_id"] for span in json.loads(spans_json)}
                    all_parts.update(parts)
                    if status == "succeeded":
                        verified_chunks += 1
                    else:
                        incomplete_parts.update(parts)
                await connection.execute(
                    "UPDATE voiceover_jobs SET verified_chunks=?,verified_parts=?,updated_at=? "
                    "WHERE job_id=?",
                    (verified_chunks, len(all_parts - incomplete_parts),
                     completed_at.isoformat(), identity.job_id),
                )
                await connection.commit()
                return result
            except BaseException:
                await connection.rollback()
                raise

    async def reconcile_interrupted(
        self, ownership: WorkspaceLock, reconciled_at: datetime
    ) -> RestartReconciliationResult:
        """Pause abandoned work under the database directory's exclusive lock.

        No artifact/provider I/O occurs here. Successful artifacts still require
        independent re-verification and complete sidecars require explicit adoption.
        The caller must retain ownership throughout recovery and worker execution.
        """
        database_identity = ownership.require_database(Path(self.db_path))
        if reconciled_at.tzinfo is None or reconciled_at.utcoffset() is None:
            raise ValueError("reconciled_at must be timezone-aware")
        timestamp = reconciled_at.astimezone(UTC).isoformat()
        paused: list[str] = []
        uncertain: list[str] = []
        abandoned: list[str] = []
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT job_id FROM voiceover_jobs j
                    WHERE deleted_at IS NULL AND status NOT IN ('completed', 'cancelled')
                      AND (status IN ('queued', 'running', 'assembling')
                        OR EXISTS (SELECT 1 FROM generation_attempts a
                          WHERE a.job_id=j.job_id AND a.dispatch_state IN ('reserved','dispatched'))
                        OR EXISTS (SELECT 1 FROM voiceover_chunks c
                          WHERE c.job_id=j.job_id AND c.status='generating'))
                    ORDER BY job_id
                    """,
                ) as cursor:
                    jobs = await cursor.fetchall()
                for (job_id,) in jobs:
                    async with connection.execute(
                        "SELECT attempt_id,chunk_id,dispatch_state FROM generation_attempts "
                        "WHERE job_id=? AND dispatch_state IN ('reserved','dispatched') "
                        "ORDER BY attempt_id", (job_id,),
                    ) as cursor:
                        attempts = await cursor.fetchall()
                    for attempt_id, chunk_id, state in attempts:
                        unknown = state == "dispatched"
                        (uncertain if unknown else abandoned).append(attempt_id)
                        reason = "UPSTREAM_OUTCOME_UNKNOWN" if unknown else "RESTART_ABANDONED_RESERVATION"
                        await connection.execute(
                            "UPDATE generation_attempts SET dispatch_state=?,ended_at=?,"
                            "sanitized_outcome=?,outcome_unknown=? WHERE attempt_id=?",
                            ("unknown" if unknown else "cancelled", timestamp, reason,
                             int(unknown), attempt_id),
                        )
                        if unknown:
                            await connection.execute(
                                "UPDATE voiceover_chunks SET status='unknown',latest_error=? "
                                "WHERE job_id=? AND chunk_id=? AND status!='succeeded'",
                                (reason, job_id, chunk_id),
                            )
                    # A generating chunk without a ledger row is also unsafe to replay.
                    await connection.execute(
                        "UPDATE voiceover_chunks SET status='unknown',"
                        "latest_error='UPSTREAM_OUTCOME_UNKNOWN' WHERE job_id=? AND status='generating'",
                        (job_id,),
                    )
                    async with connection.execute(
                        "SELECT 1 FROM voiceover_chunks WHERE job_id=? AND status='unknown' LIMIT 1",
                        (job_id,),
                    ) as cursor:
                        has_unknown = await cursor.fetchone() is not None
                    await connection.execute(
                        "UPDATE voiceover_jobs SET status='paused',reason=?,updated_at=? WHERE job_id=?",
                        ("UPSTREAM_OUTCOME_UNKNOWN" if has_unknown else "PROCESS_RESTARTED", timestamp, job_id),
                    )
                    paused.append(job_id)
                if ownership.require_database(Path(self.db_path)) != database_identity:
                    raise WorkspaceOwnershipError("Database changed during reconciliation")
                result = RestartReconciliationResult(
                    paused_job_ids=tuple(paused), uncertain_attempt_ids=tuple(uncertain),
                    abandoned_reservation_ids=tuple(abandoned),
                )
                await connection.commit()
                return result
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
