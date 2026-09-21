from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from elevenlabs_mcp.artifacts import ArtifactIdentity, ArtifactVerifier


def _source(root: Path, payload: bytes):
    identity = ArtifactIdentity(
        job_id="job",
        chunk_id="chunk",
        attempt_id="attempt",
        generation_fingerprint="sha256:" + "a" * 64,
    )
    path = root / "jobs/job/chunks/chunk/attempt.mp3"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    path.with_suffix(".complete.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "identity": identity.model_dump(),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            }
        )
    )
    return identity, path


@pytest.fixture
def mp3():
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25",
            "-f",
            "mp3",
            "-codec:a",
            "libmp3lame",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    return result.stdout


def test_real_mp3_is_decoded_with_measured_duration(tmp_path, mp3):
    from elevenlabs_mcp.audio import AudioVerifier

    identity, _ = _source(tmp_path, mp3)
    result = AudioVerifier(ArtifactVerifier(tmp_path)).verify(identity)
    assert 240 <= result.duration_ms <= 350
    assert result.source_codec == "mp3"
    assert result.integrity.sha256 == "sha256:" + hashlib.sha256(mp3).hexdigest()


@pytest.mark.parametrize("payload", [b"not audio", b"ID3\x04\x00\x00\x00\x00\x00\x00"])
def test_checksum_valid_but_undecodable_bytes_are_rejected(tmp_path, payload):
    from elevenlabs_mcp.audio import AudioVerificationError, AudioVerifier

    identity, _ = _source(tmp_path, payload)
    with pytest.raises(AudioVerificationError):
        AudioVerifier(ArtifactVerifier(tmp_path)).verify(identity)


def test_duration_limit_rejects_instead_of_silently_trimming(tmp_path, mp3):
    from elevenlabs_mcp.audio import AudioVerificationError, AudioVerifier

    identity, _ = _source(tmp_path, mp3)
    with pytest.raises(AudioVerificationError, match="duration"):
        AudioVerifier(ArtifactVerifier(tmp_path), max_duration_seconds=0.1).verify(
            identity
        )


def test_decode_uses_verified_snapshot_when_managed_path_changes(tmp_path, mp3):
    from elevenlabs_mcp.audio import AudioVerifier, FFmpegDecoder

    identity, path = _source(tmp_path, mp3)

    class ReplacingDecoder:
        def check_dependencies(self):
            pass

        def decode(self, source, output, *, duration_limit, timeout):
            path.write_bytes(b"changed after integrity check")
            FFmpegDecoder().decode(
                source, output, duration_limit=duration_limit, timeout=timeout
            )

    result = AudioVerifier(
        ArtifactVerifier(tmp_path), decoder=ReplacingDecoder()
    ).verify(identity)
    assert result.integrity.sha256 == "sha256:" + hashlib.sha256(mp3).hexdigest()
    assert result.duration_ms > 0


def test_missing_decoder_is_reported_before_artifact_reads(tmp_path):
    from elevenlabs_mcp.audio import (
        AudioVerificationError,
        AudioVerifier,
        FFmpegDecoder,
    )

    verifier = AudioVerifier(
        ArtifactVerifier(tmp_path),
        decoder=FFmpegDecoder(executable="/nonexistent/ffmpeg"),
    )
    with pytest.raises(AudioVerificationError, match="unavailable"):
        verifier.check_dependencies()


def test_decoder_timeout_is_sanitized(tmp_path, mp3, monkeypatch):
    from elevenlabs_mcp.audio import AudioVerificationError, AudioVerifier

    identity, _ = _source(tmp_path, mp3)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("PRIVATE_PATH", 1, stderr=b"PRIVATE_DATA")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(AudioVerificationError) as caught:
        AudioVerifier(ArtifactVerifier(tmp_path)).verify(identity)
    assert "PRIVATE" not in str(caught.value)


def test_decoder_process_is_confined_and_bounded(tmp_path, monkeypatch):
    from elevenlabs_mcp.audio import FFmpegDecoder

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with (
        (tmp_path / "source").open("w+b") as source,
        (tmp_path / "pcm").open("w+b") as output,
    ):
        FFmpegDecoder().decode(source, output, duration_limit=2, timeout=3)
    command, kwargs = calls[0]
    assert command[command.index("-protocol_whitelist") + 1] == "pipe"
    assert command[command.index("-i") + 1] == "pipe:0"
    assert command[command.index("-f") + 1] == "mp3"
    assert "-xerror" in command
    assert kwargs["timeout"] == 3
    assert kwargs.get("shell", False) is False
