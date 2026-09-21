"""Managed artifact byte integrity; decoding and database adoption are separate gates."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
_MARKER_LIMIT = 65_536
_READ_BLOCK = 1_048_576


class ArtifactIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_id: str = Field(pattern=_ID_PATTERN)
    chunk_id: str = Field(pattern=_ID_PATTERN)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    generation_fingerprint: str = Field(pattern=_HASH_PATTERN)


class CompletionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(pattern=r"^1$")
    identity: ArtifactIdentity
    sha256: str = Field(pattern=_HASH_PATTERN)
    byte_size: int = Field(ge=1)


class ArtifactIntegrityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    identity: ArtifactIdentity
    relative_path: str
    sha256: str
    byte_size: int


class ArtifactVerificationError(RuntimeError):
    """Missing, unsafe, incomplete, changed, or corrupt managed source bytes."""


class ArtifactVerifier:
    def __init__(self, output_root: Path, max_source_bytes: int = 268_435_456) -> None:
        if type(max_source_bytes) is not int or max_source_bytes < 1:
            raise ValueError("max_source_bytes must be a positive integer")
        self.output_root = output_root
        self.max_source_bytes = max_source_bytes

    def verify_complete(self, identity: ArtifactIdentity) -> ArtifactIntegrityResult:
        """Verify an attempt's completion marker and bytes without trusting path strings.

        The result describes this read only. Callers must verify again before later
        reuse; it is neither a decode result nor permission to skip recovery checks.
        Platforms without no-follow, descriptor-relative opens fail explicitly.
        """
        if (
            not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
        ):
            raise ArtifactVerificationError("Secure artifact reads are unavailable")
        try:
            # Validate again even if a caller used model_copy/model_construct.
            identity = ArtifactIdentity.model_validate(identity.model_dump())
            parts = ("jobs", identity.job_id, "chunks", identity.chunk_id)
            name = identity.attempt_id + ".mp3"
            with self._directory(parts) as directory:
                with _regular_file(
                    directory, identity.attempt_id + ".complete.json"
                ) as marker:
                    before = os.fstat(marker)
                    if before.st_size > _MARKER_LIMIT:
                        raise ArtifactVerificationError(
                            "Completion marker exceeds size limit"
                        )
                    marker_bytes = _read_limited(marker, _MARKER_LIMIT)
                    if _signature(before) != _signature(os.fstat(marker)):
                        raise ArtifactVerificationError(
                            "Completion marker changed during verification"
                        )
                record = CompletionRecord.model_validate_json(marker_bytes)
                if record.identity != identity:
                    raise ArtifactVerificationError(
                        "Completion marker ownership or fingerprint mismatch"
                    )
                if record.byte_size > self.max_source_bytes:
                    raise ArtifactVerificationError(
                        "Source artifact exceeds size limit"
                    )
                with _regular_file(directory, name) as source:
                    before = os.fstat(source)
                    if before.st_size != record.byte_size:
                        raise ArtifactVerificationError("Source artifact size mismatch")
                    digest = hashlib.sha256()
                    count = 0
                    while count <= record.byte_size:
                        block = os.read(
                            source, min(_READ_BLOCK, record.byte_size + 1 - count)
                        )
                        if not block:
                            break
                        count += len(block)
                        digest.update(block)
                    if (
                        count != record.byte_size
                        or "sha256:" + digest.hexdigest() != record.sha256
                        or _signature(before) != _signature(os.fstat(source))
                    ):
                        raise ArtifactVerificationError(
                            "Source artifact changed or checksum mismatch"
                        )
            return ArtifactIntegrityResult(
                identity=identity,
                relative_path="/".join((*parts, name)),
                sha256=record.sha256,
                byte_size=record.byte_size,
            )
        except (OSError, ValidationError):
            raise ArtifactVerificationError(
                "Artifact is missing, unsafe, or has an invalid completion marker"
            ) from None

    @contextmanager
    def _directory(self, parts: tuple[str, ...]) -> Iterator[int]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open(self.output_root, flags)
        try:
            for part in parts:
                child = os.open(part, flags, dir_fd=directory)
                os.close(directory)
                directory = child
            yield directory
        finally:
            os.close(directory)


@contextmanager
def _regular_file(directory: int, name: str) -> Iterator[int]:
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactVerificationError("Artifact must be a regular file")
        yield descriptor
    finally:
        os.close(descriptor)


def _signature(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_limited(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    count = 0
    while count <= limit:
        block = os.read(descriptor, min(_READ_BLOCK, limit + 1 - count))
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        count += len(block)
    raise ArtifactVerificationError("Completion marker exceeds size limit")
