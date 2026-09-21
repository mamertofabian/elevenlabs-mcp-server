"""Ordered, bounded PCM assembly and verified artifact retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path, PurePosixPath
from tempfile import TemporaryFile
from uuid import uuid4

from .artifacts import (
    ArtifactIdentity,
    ArtifactVerifier,
    _managed_directory,
    _regular_file,
)
from .audio import FFmpegDecoder


class AssemblyError(RuntimeError):
    pass


def read_artifact(
    root: Path, record: dict, *, inline: bool = False, limit: int = 8 * 1024 * 1024
):
    path = record["relative_path"]
    parts = PurePosixPath(path).parts
    if (
        not parts
        or parts[0] != "jobs"
        or any(p in {"..", "."} or "\\" in p or ":" in p for p in parts)
        or PurePosixPath(path).is_absolute()
    ):
        raise AssemblyError("PATH_NOT_ALLOWED")
    if inline and record["byte_size"] > limit:
        raise AssemblyError("ARTIFACT_TOO_LARGE")
    digest = hashlib.sha256()
    count = 0
    blocks = []
    try:
        with (
            _managed_directory(root, parts[:-1]) as directory,
            _regular_file(directory, parts[-1]) as fd,
        ):
            if os.fstat(fd).st_size != record["byte_size"]:
                raise AssemblyError("ARTIFACT_CORRUPT")
            while count <= record["byte_size"]:
                block = os.read(fd, min(1024 * 1024, record["byte_size"] + 1 - count))
                if not block:
                    break
                digest.update(block)
                count += len(block)
                if inline:
                    blocks.append(block)
        if (
            count != record["byte_size"]
            or "sha256:" + digest.hexdigest() != record["sha256"]
        ):
            raise AssemblyError("ARTIFACT_CORRUPT")
    except OSError:
        raise AssemblyError("ARTIFACT_NOT_FOUND") from None
    return b"".join(blocks) if inline else str(root.resolve() / path)


class AudioAssembler:
    def __init__(self, root: Path, max_seconds: int = 7200):
        self.root = root
        self.max_seconds = max_seconds
        self.verifier = ArtifactVerifier(root)
        self.decoder = FFmpegDecoder()

    def check_dependencies(self):
        self.decoder.check_dependencies()
        # Exercise both the decoder and final encoder before chargeable work.
        with TemporaryFile("w+b") as source, TemporaryFile("w+b") as output:
            source.write(b"\0" * 882)
            source.seek(0)
            self._encode(source, output, "mp3")
            output.seek(0)
            with TemporaryFile("w+b") as proof:
                self.decoder.decode(output, proof, duration_limit=1, timeout=10)
                if proof.seek(0, 2) == 0:
                    raise AssemblyError("DEPENDENCY_MISSING")

    def _encode(self, pcm, output, format):
        pcm.seek(0)
        if format == "wav":
            with wave.open(output, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(44100)
                while block := pcm.read(1024 * 1024):
                    writer.writeframesraw(block)
            return
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "pipe",
                    "-f",
                    "s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    "-f",
                    "mp3",
                    "pipe:1",
                ],
                stdin=pcm,
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=300,
                check=False,
            )
            if result.returncode:
                raise AssemblyError("ASSEMBLY_FAILED")
        except (OSError, subprocess.TimeoutExpired):
            raise AssemblyError("DEPENDENCY_MISSING") from None

    def assemble(self, job_id: str, plan, chunks: list[dict]):
        with TemporaryFile("w+b") as pcm, TemporaryFile("w+b") as encoded:
            for chunk in chunks:
                identity = ArtifactIdentity(
                    job_id=job_id,
                    chunk_id=chunk["chunk_id"],
                    attempt_id=chunk["attempt_id"],
                    generation_fingerprint=chunk["generation_fingerprint"],
                )
                with (
                    self.verifier.verified_snapshot(identity) as (checked, source),
                    TemporaryFile("w+b") as decoded,
                ):
                    if (
                        checked.sha256 != chunk["sha256"]
                        or checked.byte_size != chunk["byte_size"]
                    ):
                        raise AssemblyError("ARTIFACT_CORRUPT")
                    self.decoder.decode(source, decoded, duration_limit=600, timeout=60)
                    size = decoded.seek(0, 2)
                    if size == 0 or size > 600 * 88200:
                        raise AssemblyError("ARTIFACT_CORRUPT")
                    decoded.seek(0)
                    shutil.copyfileobj(decoded, pcm, length=1024 * 1024)
                pcm.write(b"\0" * (chunk["pause_after_ms"] * 44100 // 1000 * 2))
                if pcm.tell() > self.max_seconds * 88200:
                    raise AssemblyError("LIMIT_EXCEEDED")
            duration = round(pcm.tell() / 88.2)
            self._encode(pcm, encoded, plan.resolved_options.export_format)
            encoded.flush()
            encoded.seek(0)
            if plan.resolved_options.export_format == "mp3":
                with TemporaryFile("w+b") as proof:
                    self.decoder.decode(
                        encoded, proof, duration_limit=self.max_seconds, timeout=300
                    )
                    decoded_ms = proof.seek(0, 2) / 88.2
                    if abs(decoded_ms - duration) > 200 or decoded_ms <= 0:
                        raise AssemblyError("ASSEMBLY_FAILED")
            else:
                with wave.open(encoded, "rb") as proof:
                    if abs(proof.getnframes() / 44.1 - duration) > 1:
                        raise AssemblyError("ASSEMBLY_FAILED")
            encoded.seek(0)
            artifact_id = str(uuid4())
            meta_id = str(uuid4())
            filename = artifact_id + "." + plan.resolved_options.export_format
            records = []
            with _managed_directory(
                self.root, ("jobs", job_id, "final"), create=True
            ) as directory:
                records.append(
                    self._publish(
                        directory,
                        job_id,
                        filename,
                        encoded,
                        artifact_id,
                        "audio/mpeg" if filename.endswith("mp3") else "audio/wav",
                        duration,
                        plan.resolved_options.export_format,
                    )
                )
                production = {
                    "schema_version": "1",
                    "job_id": job_id,
                    "plan_hash": plan.plan_hash,
                    "planner_version": plan.planner_version,
                    "total_parts": plan.total_parts,
                    "completed_parts": plan.total_parts,
                    "total_chunks": len(chunks),
                    "completed_chunks": len(chunks),
                    "duration_ms": duration,
                    "source_codec": "mp3",
                    "options": plan.resolved_options.model_dump(mode="json"),
                    "chunks": [
                        {
                            "chunk_id": c["chunk_id"],
                            "attempt_id": c["attempt_id"],
                            "provider_request_id": c["provider_request_id"],
                            "artifact_id": c["successful_artifact_id"],
                            "sha256": c["sha256"],
                            "pause_after_ms": c["pause_after_ms"],
                            "duration_ms": c["duration_ms"],
                            "character_count": c["character_count"],
                            "voice_ids": json.loads(c["voice_ids_json"]),
                            "source_spans": json.loads(c["source_spans_json"]),
                        }
                        for c in chunks
                    ],
                    "final_audio": records[0],
                    "warnings": list(plan.warnings),
                }
                with TemporaryFile("w+b") as metadata:
                    metadata.write(json.dumps(production, sort_keys=True).encode())
                    metadata.seek(0)
                    records.append(
                        self._publish(
                            directory,
                            job_id,
                            meta_id + ".production.json",
                            metadata,
                            meta_id,
                            "application/json",
                            None,
                            None,
                        )
                    )
            # Validate the exact published bytes before the completion transaction.
            for record in records:
                read_artifact(self.root, record)
            return records

    def _publish(
        self, directory, job_id, name, source, artifact_id, mime, duration, codec
    ):
        temp = "." + str(uuid4()) + ".tmp"
        digest = hashlib.sha256()
        size = 0
        fd = os.open(
            temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        try:
            with os.fdopen(fd, "wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            os.link(
                temp,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
            os.fsync(directory)
        finally:
            os.unlink(temp, dir_fd=directory)
        return {
            "artifact_id": artifact_id,
            "job_id": job_id,
            "relative_path": f"jobs/{job_id}/final/{name}",
            "sha256": "sha256:" + digest.hexdigest(),
            "byte_size": size,
            "mime_type": mime,
            "duration_ms": duration,
            "codec": codec,
        }
