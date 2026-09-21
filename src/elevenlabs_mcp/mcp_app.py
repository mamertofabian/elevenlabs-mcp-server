"""Typed MCP revival tools and shared-core legacy adapters."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Literal
from uuid import UUID, uuid4

from mcp import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import database as errors
from .assembly import AssemblyError
from .audio import AudioDependencyError
from .contracts import Script, VoiceoverOptions
from .planner import PlanningLimitError, TextFragmentationError
from .provider import ProviderError
from .workspace import WorkspaceBusyError, WorkspaceOwnershipError


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Budget(Input):
    max_total_characters: int = Field(gt=0, le=1_000_000_000)
    max_total_requests: int = Field(gt=0, le=1_000_000)


class PlanInput(Input):
    script: Script
    options: VoiceoverOptions


class SubmitInput(PlanInput):
    plan_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1, max_length=128)
    budget: Budget


class JobInput(Input):
    job_id: str = Field(
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )


class CancelInput(JobInput):
    expected_revision: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=128)


class ResumeInput(CancelInput):
    budget: Budget
    retry_uncertain: bool = False


class ListInput(Input):
    limit: int = Field(default=25, ge=1, le=100)
    cursor: str | None = None
    status: (
        Literal[
            "queued",
            "running",
            "assembling",
            "completed",
            "failed",
            "paused",
            "cancelled",
        ]
        | None
    ) = None


class ArtifactInput(Input):
    artifact_id: str = Field(
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )
    mode: Literal["metadata", "inline", "file"] = "metadata"


class VoicesInput(Input):
    query: str = Field(default="", max_length=200)
    limit: int = Field(default=25, ge=1, le=100)
    cursor: str | None = None


DEFINITIONS = {
    "plan_voiceover": (
        PlanInput,
        "Preview a deterministic script plan without spending.",
    ),
    "submit_voiceover": (SubmitInput, "Persist and run a bounded voiceover job."),
    "get_job": (JobInput, "Inspect durable job state and exposure."),
    "list_jobs": (ListInput, "List jobs without script contents."),
    "resume_voiceover": (
        ResumeInput,
        "Explicitly resume remaining work using verified artifacts.",
    ),
    "cancel_voiceover": (CancelInput, "Request cancellation of further scheduling."),
    "get_artifact": (ArtifactInput, "Retrieve an owned, integrity-checked artifact."),
    "search_voices": (VoicesInput, "Search provider voice metadata."),
    "list_models": (Input, "List models and supported renderer profiles."),
}


class RevivalTools:
    def __init__(self, service, legacy=None):
        self.service = service
        self.legacy = legacy

    def tools(self):
        return [
            types.Tool(
                name=name,
                description=description,
                input_schema=model.model_json_schema(),
            )
            for name, (model, description) in DEFINITIONS.items()
        ]

    def handles(self, name):
        return (
            name in DEFINITIONS
            or self.legacy is not None
            and name
            in {
                "generate_audio_simple",
                "generate_audio_script",
                "get_audio_file",
                "get_voiceover_history",
                "delete_job",
                "list_voices",
            }
        )

    async def call(self, name, arguments):
        try:
            if name not in DEFINITIONS:
                return await self._legacy(name, arguments)
            model = DEFINITIONS[name][0].model_validate(arguments)
            args = model.model_dump(mode="json")
            if name == "plan_voiceover":
                data = self.service.plan(**args)
            elif name == "submit_voiceover":
                data = await self.service.submit(**args)
            elif name == "get_job":
                data = await self.service.get_job(**args)
            elif name == "list_jobs":
                data = await self.service.list_jobs(**args)
            elif name == "cancel_voiceover":
                data = await self.service.cancel(**args)
            elif name == "resume_voiceover":
                data = await self.service.resume(**args)
            elif name == "get_artifact":
                data = await self.service.get_artifact(**args)
            elif name == "search_voices":
                data = await asyncio.to_thread(
                    self.service.provider.search_voices, **args
                )
            else:
                data = await asyncio.to_thread(self.service.provider.list_models)
            content = []
            if (
                isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], bytes)
            ):
                payload = data.pop("data")
                content = [
                    types.EmbeddedResource(
                        type="resource",
                        resource=types.BlobResourceContents(
                            uri=data["uri"],
                            mime_type=data["mime_type"],
                            blob=base64.b64encode(payload).decode(),
                        ),
                    )
                ]
            data = await self._wire(name, data)
            envelope = {"ok": True, "data": data}
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=json.dumps(envelope)),
                    *content,
                ],
                structured_content=envelope,
            )
        except Exception as error:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
            code = self._code(error)
            envelope = {
                "ok": False,
                "error": {
                    "code": code,
                    "message": "Owned-file cleanup failed; retry delete_job with the same job ID."
                    if str(error) == "CLEANUP_FAILED"
                    else code.replace("_", " ").capitalize() + ".",
                    "retryable": False,
                },
            }
            if (
                isinstance(arguments.get("job_id"), str)
                and len(arguments["job_id"]) == 36
            ):
                try:
                    envelope["error"]["job_id"] = str(UUID(arguments["job_id"]))
                except ValueError:
                    pass
            if code == "ARTIFACT_TOO_LARGE":
                envelope["error"]["message"] = (
                    "Artifact exceeds the inline limit; use get_artifact with metadata or file mode."
                )
            return types.CallToolResult(
                is_error=True,
                content=[types.TextContent(type="text", text=json.dumps(envelope))],
                structured_content=envelope,
            )

    async def _wire(self, name, data):
        if name == "plan_voiceover":
            result = {
                k: v
                for k, v in data.items()
                if k not in {"planned_requests", "effective_limits"}
            }
            result["schema_version"] = "1"
            result["planned_generation_requests"] = data["planned_requests"]
            result["effective_request_limits"] = {
                k: data["effective_limits"][k]
                for k in ("max_text_characters", "max_unique_voices")
            }
            if result["resolved_options"].get("voice_settings") is None:
                result["resolved_options"].pop("voice_settings", None)
            else:
                result["resolved_options"]["voice_settings"] = {
                    k: v
                    for k, v in result["resolved_options"]["voice_settings"].items()
                    if v is not None
                }
            return result
        if name in {
            "submit_voiceover",
            "get_job",
            "resume_voiceover",
            "cancel_voiceover",
        }:
            return await self._wire_job(data)
        if name == "list_jobs":
            return {
                "jobs": [await self._wire_job(job) for job in data["jobs"]],
                "next_cursor": data["next_cursor"],
            }
        if name == "get_artifact":
            record = await self.service.store.artifact(data["artifact_id"])
            view = self._wire_artifact(record)
            if "path" in data:
                view["local_path"] = data["path"]
            return view
        if name == "search_voices":
            return {
                "voices": [
                    {
                        k: v[k]
                        for k in ("voice_id", "name", "category", "preview_url")
                        if k in v and v[k] is not None
                    }
                    for v in data["voices"]
                ],
                "next_cursor": data["next_cursor"],
                "cache": "fresh",
                "warnings": [],
            }
        return data

    def _wire_artifact(self, record):
        return {
            "schema_version": "1",
            "artifact_id": record["artifact_id"],
            "job_id": record["job_id"],
            "chunk_id": record["chunk_id"],
            "kind": "chunk"
            if record["chunk_id"]
            else "production_manifest"
            if record["mime_type"] == "application/json"
            else "final",
            "uri": "voiceover://artifacts/" + record["artifact_id"],
            "sha256": record["sha256"].removeprefix("sha256:"),
            "byte_size": record["byte_size"],
            "mime_type": record["mime_type"],
            "duration_ms": record["duration_ms"],
            "source_codec": "mp3" if record["mime_type"].startswith("audio/") else None,
            "verified": True,
        }

    async def _wire_job(self, data):
        row = await self.service.store.row(data["job_id"])
        options = json.loads(row["options_json"])
        reason = data["reason"]
        normal_reasons = {None, "PROCESS_RESTARTED", "PROCESS_STOPPED"}
        error = (
            None
            if reason in normal_reasons
            else {
                "code": self._code(ValueError(reason)),
                "message": reason.replace("_", " ").capitalize() + ".",
                "retryable": False,
            }
        )
        return {
            "schema_version": "1",
            "job_id": data["job_id"],
            "origin": data["source"],
            "status": data["status"],
            "reason": data.get("warnings", [reason])[0]
            if data.get("warnings")
            else reason,
            "revision": data["revision"],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "plan_hash": data["plan_hash"],
            "engine": options["engine"],
            "model_id": options["model_id"],
            "total_parts": data["total_parts"],
            "completed_parts": data["verified_parts"],
            "total_chunks": data["total_chunks"],
            "completed_chunks": data["verified_chunks"],
            "unknown_chunks": data["unknown_chunk_ids"],
            "resumable": data["resumable"],
            "requires_uncertain_retry_ack": bool(data["unknown_chunk_ids"]),
            "cancel_requested": data["cancel_requested"],
            "budget": {
                k: data[k] for k in ("max_total_characters", "max_total_requests")
            },
            "request_ledger": {
                k: data[k] for k in ("reserved_characters", "reserved_requests")
            },
            "artifacts": [
                self._wire_artifact(a)
                for a in await self.service.store.artifacts(
                    data["job_id"], include_chunks=True
                )
            ],
            "error": error,
        }

    def _code(self, error):
        if str(error) == "INVALID_CURSOR":
            return "INVALID_SCRIPT"
        mapping = {
            AudioDependencyError: "DEPENDENCY_MISSING",
            ValidationError: "INVALID_SCRIPT",
            PlanningLimitError: "LIMIT_EXCEEDED",
            TextFragmentationError: "LIMIT_EXCEEDED",
            errors.JobNotFoundError: "JOB_NOT_FOUND",
            errors.JobBusyError: "JOB_BUSY",
            errors.RevisionConflictError: "REVISION_CONFLICT",
            errors.IdempotencyConflictError: "IDEMPOTENCY_CONFLICT",
            errors.BudgetExceededError: "BUDGET_EXCEEDED",
            errors.UncertainAttemptError: "UPSTREAM_OUTCOME_UNKNOWN",
            WorkspaceBusyError: "WORKSPACE_BUSY",
            WorkspaceOwnershipError: "WORKSPACE_BUSY",
        }
        for kind, code in mapping.items():
            if isinstance(error, kind):
                return code
        if isinstance(error, ProviderError):
            return (
                error.code
                if error.code
                in {
                    "UNSUPPORTED_MODEL",
                    "UNSUPPORTED_SETTINGS",
                    "API_KEY_MISSING",
                    "AUTHENTICATION_FAILED",
                    "PROVIDER_VALIDATION_FAILED",
                    "RATE_LIMITED",
                    "UPSTREAM_OUTCOME_UNKNOWN",
                }
                else "INTERNAL_ERROR"
            )
        allowed = {
            "PLAN_MISMATCH",
            "INVALID_SCRIPT",
            "ARTIFACT_CORRUPT",
            "ARTIFACT_NOT_FOUND",
            "ARTIFACT_TOO_LARGE",
            "PATH_NOT_ALLOWED",
            "CREDENTIAL_CONTEXT_CHANGED",
            "JOB_DELETED",
            "DEPENDENCY_MISSING",
            "ASSEMBLY_FAILED",
            "UPSTREAM_OUTCOME_UNKNOWN",
            "BUDGET_EXCEEDED",
            "AUTHENTICATION_FAILED",
            "PROVIDER_VALIDATION_FAILED",
            "RATE_LIMITED",
        }
        return str(error) if str(error) in allowed else "INTERNAL_ERROR"

    async def read_resource(self, uri):
        artifact_id = str(uri).rsplit("/", 1)[-1]
        result = await self.call(
            "get_artifact", {"artifact_id": artifact_id, "mode": "inline"}
        )
        if result.is_error:
            raise ValueError(result.content[0].text)
        blob = result.content[1].resource
        return types.ReadResourceResult(contents=[blob])

    async def _history(self, job_id=None):
        assert self.legacy is not None
        rows = []
        ids = [job_id] if job_id else await self.service.store.ids()
        for identity in ids:
            try:
                row = await self.service.store.row(identity)
            except errors.JobNotFoundError:
                continue
            plan = json.loads(row["plan_json"])
            script = plan["normalized_script"]
            original_parts = await self.service.store.legacy_parts(identity)
            output_file = None
            for artifact in await self.service.store.artifacts(identity):
                if artifact["mime_type"].startswith("audio/"):
                    try:
                        output_file = (
                            await self.service.get_artifact(
                                artifact["artifact_id"], "file"
                            )
                        )["path"]
                    except AssemblyError:
                        row["status"] = "failed"
                        row["reason"] = "ARTIFACT_CORRUPT"
                    break
            rows.append(
                {
                    "id": identity,
                    "status": {
                        "queued": "pending",
                        "running": "processing",
                        "assembling": "processing",
                        "paused": "failed",
                        "cancelled": "failed",
                    }.get(row["status"], row["status"]),
                    "script_parts": original_parts
                    if original_parts is not None
                    else [
                        {
                            "text": p["text"],
                            "actor": p["actor"],
                            "voice_id": script["cast"][p["actor"]]["voice_id"],
                        }
                        for scene in script["scenes"]
                        for p in scene["parts"]
                    ],
                    "output_file": output_file,
                    "error": row["reason"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "total_parts": row["total_parts"],
                    "completed_parts": row["verified_parts"],
                }
            )
        legacy_rows = (
            [await self.legacy.db.get_job(job_id)]
            if job_id
            else await self.legacy.db.get_all_jobs()
        )
        rows.extend(row.to_dict() for row in legacy_rows if row)
        return rows

    async def _legacy(self, name, args):
        assert self.legacy is not None
        if name in {"generate_audio_simple", "generate_audio_script"}:
            if name == "generate_audio_script":
                parts, diagnostics = self.legacy.parse_script(args.get("script", ""))
                if any(
                    message.startswith("Skipped non-object") for message in diagnostics
                ):
                    raise ValueError("INVALID_SCRIPT")
            else:
                parts = [
                    {
                        "text": args.get("text", "").strip(),
                        "voice_id": args.get("voice_id"),
                    }
                ]
            script = {
                "script_version": "1",
                "cast": {
                    f"a{i}": {"voice_id": p.get("voice_id") or self.legacy.api.voice_id}
                    for i, p in enumerate(parts)
                },
                "scenes": [
                    {
                        "id": "scene",
                        "parts": [
                            {"id": f"p{i}", "actor": f"a{i}", "text": p["text"]}
                            for i, p in enumerate(parts)
                        ],
                    }
                ],
            }
            settings = {
                "stability": self.legacy.api.stability,
                "similarity_boost": self.legacy.api.similarity_boost,
            }
            if self.legacy.api.MODELS[self.legacy.api.model_id]["supports_style"]:
                settings["style"] = self.legacy.api.style
            options = {
                "engine": "tts",
                "model_id": self.legacy.api.model_id,
                "voice_settings": settings,
            }
            plan = self.service.plan(script, options)
            job = await self.service.submit(
                script,
                options,
                plan["plan_hash"],
                str(uuid4()),
                {
                    "max_total_characters": plan["total_characters"],
                    "max_total_requests": plan["planned_requests"],
                },
                _legacy_parts=parts,
            )
            while job["status"] in {"queued", "running", "assembling"}:
                await asyncio.sleep(0.05)
                job = await self.service.get_job(job["job_id"])
            if job["status"] != "completed":
                return types.CallToolResult(
                    is_error=True,
                    content=[
                        types.TextContent(
                            type="text",
                            text="Error generating audio. "
                            + (job["reason"] or job["status"]),
                        )
                    ],
                )
            artifact = await self.service.get_artifact(
                job["final_artifact_ids"][0], "inline"
            )
            return self._legacy_audio(artifact, "Audio generation successful.")
        if name == "get_voiceover_history":
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=json.dumps(await self._history(args.get("job_id"))),
                    )
                ]
            )
        if name == "get_audio_file":
            try:
                job = await self.service.get_job(args["job_id"])
                if not job["final_artifact_ids"]:
                    raise ValueError("ARTIFACT_NOT_FOUND")
                return self._legacy_audio(
                    await self.service.get_artifact(
                        job["final_artifact_ids"][0], "inline"
                    )
                )
            except errors.JobNotFoundError:
                return await self._legacy_file(args["job_id"])
        if name == "delete_job":
            try:
                await self.service.delete(args["job_id"])
            except errors.JobNotFoundError:
                await self._legacy_delete(args["job_id"])
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text", text="Successfully deleted job " + args["job_id"]
                    )
                ]
            )
        page = await asyncio.to_thread(
            self.service.provider.search_voices, "", 100, None
        )
        voices = page["voices"]
        while page["next_cursor"]:
            page = await asyncio.to_thread(
                self.service.provider.search_voices, "", 100, page["next_cursor"]
            )
            voices += page["voices"]
            if len(voices) > 5000:
                raise ValueError("LIMIT_EXCEEDED")
        for voice in voices:
            voice["is_default"] = voice["voice_id"] == self.legacy.api.voice_id
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(voices))]
        )

    def _legacy_audio(self, artifact, message=None):
        uri = artifact["uri"]
        if "job_id" in artifact:
            extension = "wav" if artifact["mime_type"] == "audio/wav" else "mp3"
            uri = f"audio://voiceover_{artifact['job_id']}.{extension}"
        content: list[types.ContentBlock] = (
            [types.TextContent(type="text", text=message)] if message else []
        )
        content.append(
            types.EmbeddedResource(
                type="resource",
                resource=types.BlobResourceContents(
                    uri=uri,
                    mime_type=artifact["mime_type"],
                    blob=base64.b64encode(artifact["data"]).decode(),
                ),
            )
        )
        return types.CallToolResult(content=content)

    def _legacy_path(self, output_file, job_id):
        import os
        import re
        from pathlib import Path

        assert self.legacy is not None
        path = Path(output_file)
        if not path.is_absolute():
            path = self.legacy.settings.launch_cwd / path
        try:
            relative = path.relative_to(self.service.root.absolute())
        except ValueError:
            raise ValueError("PATH_NOT_ALLOWED") from None
        if not relative.parts or any(
            part in {"..", "."} or "\\" in part or ":" in part
            for part in relative.parts
        ):
            raise ValueError("PATH_NOT_ALLOWED")
        if len(relative.parts) != 1 or len(job_id) > 128:
            raise ValueError("PATH_NOT_ALLOWED")
        if not re.fullmatch(
            r"(?:full|partial)_audio_(?:" + re.escape(job_id) + r"|[0-9]{14})\.mp3",
            relative.name,
        ):
            raise ValueError("PATH_NOT_ALLOWED")
        try:
            if os.path.samefile(path, self.service.store.db_path):
                raise ValueError("PATH_NOT_ALLOWED")
        except FileNotFoundError:
            pass
        return relative.parts

    async def _assert_legacy_reference(self, job_id, filename):
        from pathlib import Path

        async with (
            self.service.store.connect() as db,
            db.execute(
                "SELECT output_file FROM audio_jobs WHERE id!=? AND output_file IS NOT NULL",
                (job_id,),
            ) as cursor,
        ):
            if any(Path(row[0]).name == filename for row in await cursor.fetchall()):
                raise ValueError("PATH_NOT_ALLOWED")

    async def _legacy_file(self, job_id):
        import os

        from .artifacts import _managed_directory, _regular_file

        assert self.legacy is not None
        job = await self.legacy.db.get_job(job_id)
        if not job or not job.output_file:
            raise ValueError("ARTIFACT_NOT_FOUND")
        parts = self._legacy_path(job.output_file, job_id)
        await self._assert_legacy_reference(job_id, parts[-1])

        def read():
            try:
                with (
                    _managed_directory(self.service.root, parts[:-1]) as directory,
                    _regular_file(directory, parts[-1]) as descriptor,
                ):
                    info = os.fstat(descriptor)
                    if info.st_nlink != 1:
                        raise ValueError("PATH_NOT_ALLOWED")
                    size = info.st_size
                    if size > 8 * 1024 * 1024:
                        raise ValueError("ARTIFACT_TOO_LARGE")
                    blocks = []
                    count = 0
                    while count <= size:
                        block = os.read(descriptor, min(65536, size + 1 - count))
                        if not block:
                            break
                        blocks.append(block)
                        count += len(block)
                    if count != size:
                        raise ValueError("ARTIFACT_CORRUPT")
                    return b"".join(blocks)
            except OSError:
                raise ValueError("PATH_NOT_ALLOWED") from None

        data = await asyncio.to_thread(read)
        return self._legacy_audio(
            {"uri": "audio://" + parts[-1], "mime_type": "audio/mpeg", "data": data}
        )

    async def _legacy_delete(self, job_id):
        import os

        from .artifacts import _managed_directory

        assert self.legacy is not None
        job = await self.legacy.db.get_job(job_id)
        if not job:
            raise errors.JobNotFoundError(job_id)
        if job.status in {"pending", "processing"}:
            raise errors.JobBusyError(job_id)
        if job.output_file:
            parts = self._legacy_path(job.output_file, job_id)
            await self._assert_legacy_reference(job_id, parts[-1])

            def remove():
                try:
                    with _managed_directory(self.service.root, parts[:-1]) as directory:
                        if (
                            os.stat(
                                parts[-1], dir_fd=directory, follow_symlinks=False
                            ).st_nlink
                            != 1
                        ):
                            raise ValueError("PATH_NOT_ALLOWED")
                        os.unlink(parts[-1], dir_fd=directory)
                except FileNotFoundError:
                    pass
                except OSError:
                    raise ValueError("CLEANUP_FAILED") from None

            await asyncio.to_thread(remove)
        await self.legacy.db.delete_job(job_id)
