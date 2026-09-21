"""Runtime queries and finalization transactions for the durable voiceover core."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from uuid import uuid4

import aiosqlite

from .database import JobBusyError, JobNotFoundError, JobStore


class RuntimeStore(JobStore):
    async def initialize(self):
        path = Path(self.db_path)
        if path.exists():
            async with self.connect() as source:
                async with source.execute("PRAGMA user_version") as cursor:
                    version_row = await cursor.fetchone()
                    if version_row is None:
                        raise RuntimeError("SQLite did not report a schema version")
                    version = version_row[0]
                if version < 2:
                    async with source.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('audio_jobs','voices')"
                    ) as cursor:
                        tables = [r[0] for r in await cursor.fetchall()]
                    populated = False
                    for table in tables:
                        async with source.execute(
                            f"SELECT 1 FROM {table} LIMIT 1"
                        ) as cursor:
                            populated = populated or await cursor.fetchone() is not None
                    if populated:
                        backup = path.with_name(
                            path.name + ".pre-revival." + uuid4().hex + ".bak"
                        )
                        fd = os.open(
                            backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                        )
                        os.close(fd)
                        try:
                            async with aiosqlite.connect(backup) as target:
                                await source.backup(target)
                        except BaseException:
                            backup.unlink(missing_ok=True)
                            raise
        await super().initialize()
        async with self.connect() as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS runtime_schema (version INTEGER PRIMARY KEY CHECK(version=1))"
            )
            async with db.execute("SELECT version FROM runtime_schema") as cursor:
                versions = [row[0] for row in await cursor.fetchall()]
            if versions and versions != [1]:
                raise RuntimeError("Unsupported runtime schema version")
            await db.execute("INSERT OR IGNORE INTO runtime_schema VALUES(1)")
            await db.execute(
                "CREATE TABLE IF NOT EXISTS job_context (job_id TEXT PRIMARY KEY REFERENCES voiceover_jobs(job_id), credential_fingerprint TEXT NOT NULL)"
            )
            await db.execute(
                "CREATE TABLE IF NOT EXISTS legacy_render_inputs (job_id TEXT PRIMARY KEY REFERENCES voiceover_jobs(job_id), script_parts_json TEXT NOT NULL)"
            )
            await db.commit()

    async def row(self, job_id):
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM voiceover_jobs WHERE job_id=? AND deleted_at IS NULL",
                (job_id,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            return dict(row)

    async def chunks(self, job_id):
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT c.*,a.attempt_id,a.relative_path,a.sha256,a.byte_size,a.duration_ms,g.provider_request_id FROM voiceover_chunks c LEFT JOIN production_artifacts a ON c.successful_artifact_id=a.artifact_id LEFT JOIN generation_attempts g ON g.attempt_id=a.attempt_id WHERE c.job_id=? ORDER BY chunk_index",
                (job_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def artifacts(self, job_id, include_chunks=False):
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM production_artifacts WHERE job_id=? AND deleted_at IS NULL AND complete=1 "
                + ("" if include_chunks else "AND chunk_id IS NULL ")
                + "ORDER BY artifact_id",
                (job_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def artifact(self, artifact_id):
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT a.* FROM production_artifacts a JOIN voiceover_jobs j ON a.job_id=j.job_id WHERE a.artifact_id=? AND a.deleted_at IS NULL AND j.deleted_at IS NULL",
                (artifact_id,),
            ) as cur:
                r = await cur.fetchone()
                if r is None:
                    raise ValueError("ARTIFACT_NOT_FOUND")
                return dict(r)

    async def ids(self, statuses=None):
        async with self.connect() as db:
            sql = "SELECT job_id FROM voiceover_jobs WHERE deleted_at IS NULL"
            params = ()
            if statuses:
                sql += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
                params = tuple(statuses)
            sql += " ORDER BY created_at,job_id"
            async with db.execute(sql, params) as cur:
                return [r[0] for r in await cur.fetchall()]

    async def page(self, limit, cursor=None, status=None):
        parameters = []
        query = "SELECT job_id,created_at FROM voiceover_jobs WHERE deleted_at IS NULL"
        if status:
            query += " AND status=?"
            parameters.append(status)
        if cursor:
            try:
                if len(cursor) > 2048:
                    raise ValueError
                stamp, identity, filter_status = json.loads(
                    base64.urlsafe_b64decode(cursor.encode())
                )
                if (
                    not isinstance(stamp, str)
                    or not isinstance(identity, str)
                    or len(identity) > 128
                    or filter_status != status
                ):
                    raise ValueError
            except (ValueError, TypeError, UnicodeError):
                raise ValueError("INVALID_CURSOR") from None
            query += " AND (created_at,job_id)<(?,?)"
            parameters.extend((stamp, identity))
        query += " ORDER BY created_at DESC,job_id DESC LIMIT ?"
        parameters.append(limit + 1)
        async with self.connect() as db, db.execute(query, parameters) as cursor_obj:
            rows = list(await cursor_obj.fetchall())
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit:
            identity, stamp = page[-1]
            next_cursor = base64.urlsafe_b64encode(
                json.dumps([stamp, identity, status]).encode()
            ).decode()
        return [row[0] for row in page], next_cursor

    async def context(self, job_id, value=None):
        async with self.connect() as db:
            if value is not None:
                await db.execute(
                    "INSERT OR IGNORE INTO job_context VALUES(?,?)", (job_id, value)
                )
                await db.commit()
            async with db.execute(
                "SELECT credential_fingerprint FROM job_context WHERE job_id=?",
                (job_id,),
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def legacy_parts(self, job_id):
        async with (
            self.connect() as db,
            db.execute(
                "SELECT script_parts_json FROM legacy_render_inputs WHERE job_id=?",
                (job_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        return json.loads(row[0]) if row else None

    async def provider_receipt(self, attempt_id, request_id):
        if request_id is None:
            return
        async with self.connect() as db:
            await db.execute(
                "UPDATE generation_attempts SET provider_request_id=? WHERE attempt_id=?",
                (request_id, attempt_id),
            )
            await db.commit()

    async def transition(self, job_id, status, reason, now):
        async with self.connect() as db:
            await db.execute(
                "UPDATE voiceover_jobs SET status=?,reason=?,updated_at=? WHERE job_id=? AND deleted_at IS NULL",
                (status, reason, now.isoformat(), job_id),
            )
            await db.commit()

    async def has_dispatched(self, job_id):
        async with (
            self.connect() as db,
            db.execute(
                "SELECT 1 FROM generation_attempts WHERE job_id=? AND dispatch_state='dispatched' LIMIT 1",
                (job_id,),
            ) as cursor,
        ):
            return await cursor.fetchone() is not None

    async def invalidate(self, job_id, chunk_id, now):
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                "UPDATE production_artifacts SET complete=0 WHERE artifact_id=(SELECT successful_artifact_id FROM voiceover_chunks WHERE job_id=? AND chunk_id=?)",
                (job_id, chunk_id),
            )
            await db.execute(
                "UPDATE voiceover_chunks SET status='unknown',latest_error='ARTIFACT_CORRUPT' WHERE job_id=? AND chunk_id=?",
                (job_id, chunk_id),
            )
            async with db.execute(
                "SELECT status,source_spans_json FROM voiceover_chunks WHERE job_id=?",
                (job_id,),
            ) as cursor:
                chunks = await cursor.fetchall()
            all_parts = set()
            incomplete = set()
            for status, spans in chunks:
                parts = {span["part_id"] for span in json.loads(spans)}
                all_parts.update(parts)
                if status != "succeeded":
                    incomplete.update(parts)
            await db.execute(
                "UPDATE voiceover_jobs SET status='paused',reason='ARTIFACT_CORRUPT',verified_chunks=?,verified_parts=?,updated_at=? WHERE job_id=?",
                (
                    sum(status == "succeeded" for status, _ in chunks),
                    len(all_parts - incomplete),
                    now.isoformat(),
                    job_id,
                ),
            )
            await db.commit()

    async def invalidate_final(self, job_id, artifact_id, now):
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                "UPDATE production_artifacts SET complete=0 WHERE artifact_id=? AND job_id=?",
                (artifact_id, job_id),
            )
            await db.execute(
                "UPDATE voiceover_jobs SET status='failed',reason='ASSEMBLY_FAILED',updated_at=? WHERE job_id=?",
                (now.isoformat(), job_id),
            )
            await db.commit()

    async def complete(self, job_id, final, now):
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT cancel_requested,verified_chunks,total_chunks,status FROM voiceover_jobs WHERE job_id=? AND deleted_at IS NULL",
                    (job_id,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None or row[0] or row[1] != row[2] or row[3] != "assembling":
                    raise JobBusyError(job_id)
                await db.execute(
                    "UPDATE production_artifacts SET deleted_at=? WHERE job_id=? AND chunk_id IS NULL AND deleted_at IS NULL",
                    (now.isoformat(), job_id),
                )
                for item in final:
                    await db.execute(
                        "INSERT INTO production_artifacts(artifact_id,job_id,relative_path,sha256,byte_size,mime_type,codec,duration_ms,complete) VALUES(?,?,?,?,?,?,?,?,1)",
                        (
                            item["artifact_id"],
                            job_id,
                            item["relative_path"],
                            item["sha256"],
                            item["byte_size"],
                            item["mime_type"],
                            item.get("codec"),
                            item.get("duration_ms"),
                        ),
                    )
                await db.execute(
                    "UPDATE voiceover_jobs SET status='completed',reason=NULL,updated_at=? WHERE job_id=?",
                    (now.isoformat(), job_id),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def tombstone(self, workspace, job_id, now):
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT status FROM voiceover_jobs WHERE job_id=? AND deleted_at IS NULL",
                    (job_id,),
                ) as cur:
                    r = await cur.fetchone()
                if not r:
                    raise JobNotFoundError(job_id)
                if r[0] in {"queued", "running", "assembling"}:
                    raise JobBusyError(job_id)
                await db.execute(
                    "UPDATE voiceover_jobs SET deleted_at=?,revision=revision+1,"
                    "normalized_script_json='{}',options_json='{}',plan_json='{}' WHERE job_id=?",
                    (now.isoformat(), job_id),
                )
                await db.execute(
                    "UPDATE voiceover_chunks SET request_json='{}',source_spans_json='[]',"
                    "voice_ids_json='[]',generation_metadata_json=NULL WHERE job_id=?",
                    (job_id,),
                )
                await db.execute("DELETE FROM job_context WHERE job_id=?", (job_id,))
                await db.execute(
                    "DELETE FROM legacy_render_inputs WHERE job_id=?", (job_id,)
                )
                await db.execute(
                    "UPDATE production_artifacts SET deleted_at=? WHERE job_id=?",
                    (now.isoformat(), job_id),
                )
                await db.execute(
                    "UPDATE operation_receipts SET tombstone=1 WHERE workspace_id=? AND (result_ref=? OR json_extract(CASE WHEN json_valid(result_ref) THEN result_ref ELSE '{}' END,'$.job_id')=?)",
                    (workspace, job_id, job_id),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
