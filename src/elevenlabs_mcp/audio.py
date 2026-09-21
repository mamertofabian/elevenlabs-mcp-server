"""Bounded local MP3 decoding of integrity-verified source snapshots."""

from __future__ import annotations

import math
import shutil
import subprocess
from tempfile import TemporaryFile
from typing import BinaryIO, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import ArtifactIdentity, ArtifactIntegrityResult, ArtifactVerifier

_SAMPLE_RATE = 44_100
_BYTES_PER_SAMPLE = 2


class AudioVerificationError(RuntimeError):
    """Dependency, decode, or resource-policy failure with sanitized diagnostics."""


class AudioDecoder(Protocol):
    def check_dependencies(self) -> None: ...

    def decode(
        self,
        source: BinaryIO,
        output: BinaryIO,
        *,
        duration_limit: float,
        timeout: float,
    ) -> None: ...


class FFmpegDecoder:
    def __init__(self, executable: str = "ffmpeg") -> None:
        self.executable = executable

    def check_dependencies(self) -> None:
        if shutil.which(self.executable) is None:
            raise AudioVerificationError("FFmpeg is unavailable")

    def decode(
        self,
        source: BinaryIO,
        output: BinaryIO,
        *,
        duration_limit: float,
        timeout: float,
    ) -> None:
        command = [
            self.executable,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-protocol_whitelist",
            "pipe",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            # Decode beyond the limit so overlong audio is rejected, not accepted
            # as a silently trimmed prefix. The extra second bounds disk usage.
            "-t",
            str(duration_limit + 1),
            "-f",
            "s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                stdin=source,
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise AudioVerificationError(
                "Audio decoding exceeded its time limit"
            ) from None
        except OSError:
            raise AudioVerificationError("Audio decoder could not run") from None
        if result.returncode != 0:
            raise AudioVerificationError("Source audio failed strict MP3 decoding")


class AudioVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    integrity: ArtifactIntegrityResult
    source_codec: Literal["mp3"] = "mp3"
    duration_ms: int = Field(ge=1)


class AudioVerifier:
    def __init__(
        self,
        artifacts: ArtifactVerifier,
        decoder: AudioDecoder | None = None,
        max_duration_seconds: float = 600,
        timeout_seconds: float = 60,
    ) -> None:
        if (
            not math.isfinite(max_duration_seconds)
            or not 0 < max_duration_seconds <= 3600
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 600
        ):
            raise ValueError(
                "Audio duration and timeout limits must be finite and bounded"
            )
        self.artifacts = artifacts
        self.decoder = decoder if decoder is not None else FFmpegDecoder()
        self.max_duration_seconds = max_duration_seconds
        self.timeout_seconds = timeout_seconds

    def check_dependencies(self) -> None:
        """Expose a preflight check for callers before any chargeable synthesis."""
        self.decoder.check_dependencies()

    def verify(self, identity: ArtifactIdentity) -> AudioVerificationResult:
        self.check_dependencies()
        with (
            self.artifacts.verified_snapshot(identity) as (integrity, source),
            TemporaryFile(mode="w+b") as decoded,
        ):
            self.decoder.decode(
                source,
                decoded,
                duration_limit=self.max_duration_seconds,
                timeout=self.timeout_seconds,
            )
            byte_size = decoded.seek(0, 2)
        if byte_size == 0 or byte_size % _BYTES_PER_SAMPLE:
            raise AudioVerificationError("Source audio decoded to no complete samples")
        seconds = byte_size / (_SAMPLE_RATE * _BYTES_PER_SAMPLE)
        if seconds > self.max_duration_seconds:
            raise AudioVerificationError("Source audio exceeds decoded duration limit")
        return AudioVerificationResult(
            integrity=integrity,
            duration_ms=max(1, round(seconds * 1000)),
        )
