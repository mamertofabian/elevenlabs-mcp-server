"""Single-owner durable execution, explicit resume, and verified local retrieval."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .artifacts import (
    ArtifactIdentity,
    ArtifactPublisher,
    ArtifactVerificationError,
    ArtifactVerifier,
)
from .assembly import AssemblyError, AudioAssembler, read_artifact
from .audio import AudioVerifier
from .contracts import PlanningLimits, Script, VoiceoverOptions, VoiceoverPlan
from .database import (
    AttemptStateError,
    BudgetExceededError,
    JobBusyError,
    JobNotFoundError,
    ReservationStateError,
    RevisionConflictError,
)
from .planner import PlanningLimitError, ScriptPlanner
from .provider import ProviderError, validate_profile
from .recovery import ArtifactRecovery
from .runtime_store import RuntimeStore
from .workspace import WorkspaceLock


class VoiceoverService:
    def __init__(
        self,
        database_path: Path,
        output_root: Path,
        provider,
        *,
        limits: PlanningLimits | None = None,
        clock=None,
    ):
        self.store = RuntimeStore(database_path)
        self.root = output_root
        self.provider = provider
        self.limits = limits or PlanningLimits()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.verifier = AudioVerifier(ArtifactVerifier(output_root))
        self.publisher = ArtifactPublisher(output_root)
        self.assembler = AudioAssembler(output_root)
        self.owner = WorkspaceLock(database_path.parent)
        self.workspace = "local"
        self._wake = asyncio.Event()
        self._task = None
        self._stopping = False
        self._started = False
        self._operations = asyncio.Lock()
        self.recovery_report = None

    def _make_plan(self, script, options):
        script = Script.model_validate(script)
        options = VoiceoverOptions.model_validate(options)
        validate_profile(options)
        plan = ScriptPlanner().plan(script, options, self.limits)
        pauses = sum(request.chunk.pause_after_ms for request in plan.requests)
        if pauses >= self.assembler.max_seconds * 1000:
            raise PlanningLimitError(
                "max_output_pause_ms", self.assembler.max_seconds * 1000, pauses
            )
        return plan

    def plan(self, script, options):
        p = self._make_plan(script, options)
        return {
            "plan_hash": p.plan_hash,
            "planner_version": p.planner_version,
            "resolved_options": p.resolved_options.model_dump(mode="json"),
            "total_parts": p.total_parts,
            "total_characters": p.total_characters,
            "planned_requests": len(p.requests),
            "distinct_voice_count": p.distinct_voice_count,
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "index": r.chunk.index,
                    "scene_id": r.chunk.scene_id,
                    "character_count": r.chunk.character_count,
                    "voice_ids": list(r.chunk.voice_ids),
                    "source_spans": [
                        f.source_span.model_dump() for f in r.chunk.fragments
                    ],
                    "pause_after_ms": r.chunk.pause_after_ms,
                }
                for r in p.requests
            ],
            "effective_limits": p.effective_limits.model_dump(),
            "warnings": list(p.warnings),
            "provider_access_checked": False,
            "cost_estimate": None,
        }

    async def start(self):
        if self._started:
            return
        Path(self.store.db_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner.__enter__()
        try:
            if Path(self.store.db_path).exists():
                self.owner.require_database(Path(self.store.db_path))
            else:
                descriptor = os.open(
                    self.store.db_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                os.close(descriptor)
            await self.store.initialize()
            self.recovery_report = await ArtifactRecovery(
                self.store, self.verifier
            ).recover(self.owner, self.clock())
            for job_id in await self.store.ids():
                await self._verify_recorded(job_id, invalidate=True)
                row = await self.store.row(job_id)
                if row["cancel_requested"] and row["status"] in {"paused", "failed"}:
                    await self.store.transition(
                        job_id, "cancelled", row["reason"], self.clock()
                    )
            self._started = True
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="voiceover-worker")
        except BaseException:
            self.owner.__exit__(None, None, None)
            raise

    async def close(self):
        if not self._started:
            return
        self._stopping = True
        self._wake.set()
        try:
            if self._task:
                await self._task
        finally:
            try:
                self.provider.close()
            finally:
                self._started = False
                self.owner.__exit__(None, None, None)

    def _require_started(self):
        if not self._started or self._stopping:
            raise RuntimeError("WORKER_NOT_STARTED")
        if self._task is not None and self._task.done():
            raise RuntimeError("WORKER_FAILED")
        self.owner.require_database(Path(self.store.db_path))

    async def submit(
        self, script, options, plan_hash, idempotency_key, budget, *, _legacy_parts=None
    ):
        self._require_started()
        p = self._make_plan(script, options)
        if (
            _legacy_parts is not None
            and len(json.dumps(_legacy_parts).encode("utf-8")) > 1024 * 1024
        ):
            raise ValueError("LIMIT_EXCEEDED")
        if p.plan_hash != plan_hash:
            raise ValueError("PLAN_MISMATCH")
        async with self._operations:
            async with (
                self.store.connect() as db,
                db.execute(
                    "SELECT tombstone FROM operation_receipts WHERE workspace_id=? AND operation='submit' AND idempotency_key=?",
                    (self.workspace, idempotency_key),
                ) as cur,
            ):
                existing = await cur.fetchone()
            if existing is None:
                self.provider.check_ready(p.resolved_options)
                await asyncio.to_thread(self.assembler.check_dependencies)
            result = await self.store.create_job(
                self.workspace,
                str(uuid4()),
                p,
                idempotency_key,
                budget["max_total_characters"],
                budget["max_total_requests"],
                self.clock(),
                credential_context=getattr(
                    self.provider, "context_id", "injected-provider"
                ),
                legacy_script_parts=_legacy_parts,
            )
            if existing and existing[0]:
                raise ValueError("JOB_DELETED")
            if result.created:
                self._wake.set()
            return await self.get_job(result.job_id)

    async def get_job(self, job_id):
        row = await self.store.row(job_id)
        chunks = await self.store.chunks(job_id)
        artifacts = await self.store.artifacts(job_id)
        keep = (
            "job_id",
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
            "plan_hash",
            "source",
        )
        view = {k: row[k] for k in keep}
        view.update(
            cancel_requested=bool(row["cancel_requested"]),
            resumable=row["status"] in {"paused", "failed", "cancelled"}
            and not any(c["status"] == "generating" for c in chunks),
            unknown_chunk_ids=[
                c["chunk_id"] for c in chunks if c["status"] == "unknown"
            ],
            chunks=[
                {
                    k: c[k]
                    for k in (
                        "chunk_id",
                        "chunk_index",
                        "status",
                        "character_count",
                        "pause_after_ms",
                        "successful_artifact_id",
                        "latest_error",
                    )
                }
                for c in chunks
            ],
            final_artifact_ids=[
                a["artifact_id"]
                for a in artifacts
                if a["mime_type"].startswith("audio/")
            ],
            production_artifact_ids=[
                a["artifact_id"]
                for a in artifacts
                if a["mime_type"] == "application/json"
            ],
        )
        return view

    async def list_jobs(self, limit=25, cursor=None, status=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("LIMIT_EXCEEDED")
        page, next_cursor = await self.store.page(limit, cursor, status)
        views = []
        for job_id in page:
            view = await self.get_job(job_id)
            view.pop("chunks")
            views.append(view)
        return {"jobs": views, "next_cursor": next_cursor}

    async def _verify_recorded(self, job_id, *, invalidate=False):
        verified = []
        for chunk in await self.store.chunks(job_id):
            if chunk["status"] != "succeeded":
                continue
            try:
                identity = ArtifactIdentity(
                    job_id=job_id,
                    chunk_id=chunk["chunk_id"],
                    attempt_id=chunk["attempt_id"],
                    generation_fingerprint=chunk["generation_fingerprint"],
                )
                result = await asyncio.to_thread(
                    self.verifier.artifacts.verify_complete, identity
                )
                if (
                    result.sha256 != chunk["sha256"]
                    or result.byte_size != chunk["byte_size"]
                ):
                    raise ArtifactVerificationError("Mismatch")
                verified.append(chunk["successful_artifact_id"])
            except (ArtifactVerificationError, ValueError):
                if invalidate:
                    await self.store.invalidate(job_id, chunk["chunk_id"], self.clock())
                else:
                    raise ValueError("ARTIFACT_CORRUPT") from None
        row = await self.store.row(job_id)
        if row["status"] == "completed":
            artifacts = await self.store.artifacts(job_id)
            if not any(
                a["mime_type"].startswith("audio/") for a in artifacts
            ) or not any(a["mime_type"] == "application/json" for a in artifacts):
                if invalidate:
                    await self.store.transition(
                        job_id, "failed", "ASSEMBLY_FAILED", self.clock()
                    )
                else:
                    raise AssemblyError("ARTIFACT_CORRUPT")
            for artifact in artifacts:
                try:
                    await asyncio.to_thread(read_artifact, self.root, artifact)
                except AssemblyError:
                    if invalidate:
                        await self.store.invalidate_final(
                            job_id, artifact["artifact_id"], self.clock()
                        )
                    else:
                        raise
        return tuple(verified)

    async def resume(
        self, job_id, expected_revision, idempotency_key, budget, retry_uncertain=False
    ):
        self._require_started()
        async with self._operations:
            row = await self.store.row(job_id)
            # Identical mutation replay must remain side-effect-free even after subsequent work.
            async with (
                self.store.connect() as db,
                db.execute(
                    "SELECT 1 FROM operation_receipts WHERE workspace_id=? AND operation='resume' AND idempotency_key=?",
                    (self.workspace, idempotency_key),
                ) as cur,
            ):
                replay = await cur.fetchone()
            verified = ()
            if not replay:
                if row["revision"] != expected_revision:
                    raise RevisionConflictError(
                        job_id, expected_revision, row["revision"]
                    )
                if row["status"] in {"queued", "running", "assembling"}:
                    raise JobBusyError(job_id)
                await ArtifactRecovery(self.store, self.verifier).adopt_ready(
                    self.owner, self.clock(), job_id
                )
                verified = await self._verify_recorded(job_id, invalidate=True)
                if (await self.store.row(job_id))[
                    "reason"
                ] == "ARTIFACT_CORRUPT" and not retry_uncertain:
                    raise ValueError("ARTIFACT_CORRUPT")
                row = await self.store.row(job_id)
                p = VoiceoverPlan.model_validate_json(row["plan_json"])
                if len(verified) < row["total_chunks"]:
                    if await self.store.context(job_id) != getattr(
                        self.provider, "context_id", "injected-provider"
                    ):
                        raise ValueError("CREDENTIAL_CONTEXT_CHANGED")
                    self.provider.check_ready(p.resolved_options)
                if row["status"] != "completed":
                    await asyncio.to_thread(self.assembler.check_dependencies)
            result = await self.store.request_resume(
                self.workspace,
                job_id,
                expected_revision,
                idempotency_key,
                budget["max_total_characters"],
                budget["max_total_requests"],
                self.clock(),
                retry_uncertain,
                verified_artifacts=verified,
            )
            if not result.replayed and result.status == "queued":
                self._wake.set()
            view = await self.get_job(job_id)
            view["warnings"] = list(result.warnings)
            return view

    async def cancel(self, job_id, expected_revision, idempotency_key):
        self._require_started()
        async with self._operations:
            result = await self.store.request_cancel(
                self.workspace, job_id, expected_revision, idempotency_key, self.clock()
            )
            if result.replayed:
                return await self.get_job(job_id)
            row = await self.store.row(job_id)
            if row["status"] not in {
                "running",
                "assembling",
                "completed",
            } and not await self.store.has_dispatched(job_id):
                await self.store.transition(
                    job_id, "cancelled", row["reason"], self.clock()
                )
            self._wake.set()
            return await self.get_job(job_id)

    async def get_artifact(self, artifact_id, mode="metadata"):
        if mode not in {"metadata", "inline", "file"}:
            raise ValueError("INVALID_MODE")
        record = await self.store.artifact(artifact_id)
        if not record["complete"]:
            raise AssemblyError("ARTIFACT_CORRUPT")
        data = await asyncio.to_thread(
            read_artifact, self.root, record, inline=mode == "inline"
        )
        result = {
            k: record[k]
            for k in (
                "artifact_id",
                "job_id",
                "sha256",
                "byte_size",
                "mime_type",
                "duration_ms",
            )
        }
        result["uri"] = "voiceover://artifacts/" + artifact_id
        if mode == "inline":
            result["data"] = data
        if mode == "file":
            result["path"] = data
        return result

    async def delete(self, job_id):
        from .artifacts import remove_job_tree

        self._require_started()
        async with self._operations:
            async with (
                self.store.connect() as db,
                db.execute(
                    "SELECT status,deleted_at FROM voiceover_jobs WHERE job_id=?",
                    (job_id,),
                ) as cursor,
            ):
                row = await cursor.fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if row[0] in {"queued", "running", "assembling"}:
                raise JobBusyError(job_id)
            if row[1] is None:
                await self.store.tombstone(self.workspace, job_id, self.clock())
            await asyncio.to_thread(remove_job_tree, self.root, job_id)
            return {"job_id": job_id, "deleted": True}

    async def _run(self):
        while not self._stopping:
            await self._wake.wait()
            self._wake.clear()
            for job_id in await self.store.ids(["queued"]):
                if self._stopping:
                    break
                try:
                    await self._execute(job_id)
                except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                    await self.store.transition(
                        job_id, "paused", "WORKER_FAILED", self.clock()
                    )
        # No queued work is silently dispatched when a process is stopping.
        for job_id in await self.store.ids(["queued", "running", "assembling"]):
            await self.store.transition(
                job_id, "paused", "PROCESS_STOPPED", self.clock()
            )

    async def _execute(self, job_id):
        row = await self.store.row(job_id)
        p = VoiceoverPlan.model_validate_json(row["plan_json"])
        if row["verified_chunks"] < row["total_chunks"] and await self.store.context(
            job_id
        ) != getattr(self.provider, "context_id", "injected-provider"):
            await self.store.transition(
                job_id, "paused", "CREDENTIAL_CONTEXT_CHANGED", self.clock()
            )
            return
        requests = {r.chunk_id: r for r in p.requests}
        for chunk in await self.store.chunks(job_id):
            row = await self.store.row(job_id)
            if row["cancel_requested"]:
                await self.store.transition(
                    job_id, "cancelled", row["reason"], self.clock()
                )
                return
            if self._stopping:
                await self.store.transition(
                    job_id, "paused", "PROCESS_STOPPED", self.clock()
                )
                return
            if chunk["status"] == "succeeded":
                continue
            if chunk["status"] != "pending":
                raise AttemptStateError("unexpected chunk state")
            attempt_id = str(uuid4())
            request = requests[chunk["chunk_id"]]
            try:
                await self.store.reserve_attempt(
                    job_id, chunk["chunk_id"], attempt_id, self.clock()
                )
            except ReservationStateError:
                if (await self.store.row(job_id))["cancel_requested"]:
                    await self.store.transition(job_id, "cancelled", None, self.clock())
                    return
                raise
            except BudgetExceededError:
                await self.store.transition(
                    job_id, "paused", "BUDGET_EXCEEDED", self.clock()
                )
                return
            row = await self.store.row(job_id)
            if row["cancel_requested"]:
                await self.store.transition(
                    job_id, "cancelled", row["reason"], self.clock()
                )
                return
            try:
                await self.store.mark_attempt_dispatched(attempt_id, self.clock())
            except AttemptStateError:
                if (await self.store.row(job_id))["cancel_requested"]:
                    await self.store.transition(job_id, "cancelled", None, self.clock())
                    return
                raise
            identity = ArtifactIdentity(
                job_id=job_id,
                chunk_id=chunk["chunk_id"],
                attempt_id=attempt_id,
                generation_fingerprint=request.generation_fingerprint,
            )
            try:
                await asyncio.to_thread(
                    self.publisher.publish,
                    identity,
                    self.provider.generate(request, p.resolved_options),
                )
                await self.store.provider_receipt(
                    attempt_id, getattr(self.provider, "last_request_id", None)
                )
                verified = await asyncio.to_thread(self.verifier.verify, identity)
            except ProviderError as error:
                await self.store.provider_receipt(
                    attempt_id, getattr(self.provider, "last_request_id", None)
                )
                await self.store.record_attempt_failure(
                    attempt_id,
                    "unknown" if error.uncertain else "failed",
                    self.clock(),
                    getattr(self.provider, "last_request_id", None),
                )
                await self.store.transition(
                    job_id,
                    "paused" if error.uncertain else "failed",
                    error.code,
                    self.clock(),
                )
                return
            except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                # A response or publication may already exist: do not buy it again.
                await self.store.record_attempt_failure(
                    attempt_id,
                    "unknown",
                    self.clock(),
                    getattr(self.provider, "last_request_id", None),
                )
                await self.store.transition(
                    job_id, "paused", "ARTIFACT_VERIFICATION_FAILED", self.clock()
                )
                return
            await self.store.record_verified_artifact(
                str(uuid4()), verified, self.clock()
            )
        row = await self.store.row(job_id)
        if row["cancel_requested"]:
            await self.store.transition(
                job_id, "cancelled", row["reason"], self.clock()
            )
            return
        await self.store.transition(job_id, "assembling", None, self.clock())
        try:
            final = await asyncio.to_thread(
                self.assembler.assemble, job_id, p, await self.store.chunks(job_id)
            )
            await self.store.complete(job_id, final, self.clock())
        except JobBusyError:
            await self.store.transition(job_id, "cancelled", None, self.clock())
        except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
            await self.store.transition(
                job_id, "failed", "ASSEMBLY_FAILED", self.clock()
            )
