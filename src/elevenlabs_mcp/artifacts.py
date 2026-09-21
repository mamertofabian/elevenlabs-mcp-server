"""Managed artifact byte integrity; decoding and database adoption are separate gates."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import BinaryIO
from uuid import uuid4

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
        return self._verify_complete(identity)

    @contextmanager
    def verified_snapshot(
        self, identity: ArtifactIdentity
    ) -> Iterator[tuple[ArtifactIntegrityResult, BinaryIO]]:
        """Keep an owned copy of the exact verified bytes open for local decoding."""
        with TemporaryFile(mode="w+b") as snapshot:
            result = self._verify_complete(identity, snapshot)
            snapshot.seek(0)
            yield result, snapshot

    def _verify_complete(
        self, identity: ArtifactIdentity, snapshot: BinaryIO | None = None
    ) -> ArtifactIntegrityResult:
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
                        if snapshot is not None:
                            snapshot.write(block)
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
        with _managed_directory(self.output_root, parts) as directory:
            yield directory


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


class ArtifactPublicationError(RuntimeError):
    """A complete, nonempty source could not be published for this attempt."""


class ArtifactPublisher:
    def __init__(self, output_root: Path, max_source_bytes: int = 268_435_456) -> None:
        if type(max_source_bytes) is not int or max_source_bytes < 1:
            raise ValueError("max_source_bytes must be a positive integer")
        self.output_root = output_root
        self.max_source_bytes = max_source_bytes

    def publish(
        self, identity: ArtifactIdentity, chunks: Iterable[bytes]
    ) -> ArtifactIntegrityResult:
        """Publish a fully consumed provider stream, with its marker last.

        Iteration errors propagate unchanged for the provider/job layer to classify.
        A failure between the two publications may leave unmarked source bytes;
        these are intentionally not adoptable and must never be silently replaced.
        The configured root must already exist. No audio validity is implied here.
        """
        if (
            not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or any(
                operation not in os.supports_dir_fd
                for operation in (os.open, os.mkdir, os.link, os.unlink)
            )
        ):
            raise ArtifactPublicationError("Secure artifact publication is unavailable")
        identity = ArtifactIdentity.model_validate(identity.model_dump())
        parts = ("jobs", identity.job_id, "chunks", identity.chunk_id)
        audio_name = identity.attempt_id + ".mp3"
        marker_name = identity.attempt_id + ".complete.json"
        token = uuid4().hex
        audio_temp = f".{identity.attempt_id}.{token}.audio.tmp"
        marker_temp = f".{identity.attempt_id}.{token}.marker.tmp"
        owned_temps: list[str] = []
        with _managed_directory(self.output_root, parts, create=True) as directory:
            for name in (audio_name, marker_name):
                try:
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ArtifactPublicationError("Attempt artifact already exists")
            primary_error: BaseException | None = None
            try:
                descriptor = os.open(
                    audio_temp,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                owned_temps.append(audio_temp)
                digest = hashlib.sha256()
                count = 0
                with os.fdopen(descriptor, "wb") as source:
                    for block in chunks:
                        if not isinstance(block, bytes):
                            raise ArtifactPublicationError(
                                "Source stream must yield bytes"
                            )
                        count += len(block)
                        if count > self.max_source_bytes:
                            raise ArtifactPublicationError(
                                "Source artifact exceeds size limit"
                            )
                        source.write(block)
                        digest.update(block)
                    if count == 0:
                        raise ArtifactPublicationError(
                            "Source artifact must not be empty"
                        )
                    source.flush()
                    os.fsync(source.fileno())
                record = CompletionRecord(
                    schema_version="1",
                    identity=identity,
                    sha256="sha256:" + digest.hexdigest(),
                    byte_size=count,
                )
                descriptor = os.open(
                    marker_temp,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                owned_temps.append(marker_temp)
                with os.fdopen(descriptor, "wb") as marker:
                    marker.write(record.model_dump_json().encode("utf-8"))
                    marker.flush()
                    os.fsync(marker.fileno())
                # Hard links provide same-filesystem atomic no-replace publication.
                try:
                    os.link(
                        audio_temp,
                        audio_name,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                    os.fsync(directory)
                    os.link(
                        marker_temp,
                        marker_name,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                    os.fsync(directory)
                except FileExistsError:
                    raise ArtifactPublicationError(
                        "Attempt artifact already exists"
                    ) from None
                return ArtifactIntegrityResult(
                    identity=identity,
                    relative_path="/".join((*parts, audio_name)),
                    sha256=record.sha256,
                    byte_size=count,
                )
            except BaseException as error:
                primary_error = error
                raise
            finally:
                cleanup_failed = False
                for temporary in owned_temps:
                    try:
                        os.unlink(temporary, dir_fd=directory)
                    except OSError:
                        cleanup_failed = True
                if cleanup_failed:
                    if primary_error is not None:
                        primary_error.add_note("Artifact temporary-file cleanup failed")
                    else:
                        raise ArtifactPublicationError(
                            "Artifact temporary-file cleanup failed"
                        )


@contextmanager
def _managed_directory(
    root: Path, parts: tuple[str, ...], *, create: bool = False
) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=directory)
                except FileExistsError:
                    pass
                else:
                    os.fsync(directory)
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        yield directory
    finally:
        os.close(directory)
