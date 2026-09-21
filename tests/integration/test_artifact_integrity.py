from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _fixture(root: Path):
    from elevenlabs_mcp.artifacts import ArtifactIdentity

    identity = ArtifactIdentity(
        job_id="job",
        chunk_id="chunk",
        attempt_id="attempt",
        generation_fingerprint="sha256:" + "a" * 64,
    )
    audio = root / "jobs/job/chunks/chunk/attempt.mp3"
    audio.parent.mkdir(parents=True)
    payload = b"synthetic source bytes; decoding is a separate gate"
    audio.write_bytes(payload)
    marker = audio.with_suffix(".complete.json")
    record = {
        "schema_version": "1",
        "identity": identity.model_dump(),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }
    marker.write_text(json.dumps(record))
    return identity, audio, marker, record


def test_complete_matching_source_is_verified_without_modification(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactVerifier

    identity, audio, marker, record = _fixture(tmp_path)
    before = (audio.read_bytes(), marker.read_bytes())
    result = ArtifactVerifier(tmp_path).verify_complete(identity)
    assert result.relative_path == "jobs/job/chunks/chunk/attempt.mp3"
    assert result.sha256 == record["sha256"]
    assert result.byte_size == record["byte_size"]
    assert result.identity == identity
    assert (audio.read_bytes(), marker.read_bytes()) == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing_marker",
        "missing_audio",
        "truncated",
        "same_size_corruption",
        "empty",
        "bad_json",
        "oversized_marker",
    ],
)
def test_incomplete_or_corrupt_source_is_rejected(tmp_path, damage):
    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    identity, audio, marker, _ = _fixture(tmp_path)
    if damage == "missing_marker":
        marker.unlink()
    elif damage == "missing_audio":
        audio.unlink()
    elif damage == "truncated":
        audio.write_bytes(audio.read_bytes()[:-1])
    elif damage == "same_size_corruption":
        audio.write_bytes(b"x" * audio.stat().st_size)
    elif damage == "empty":
        audio.write_bytes(b"")
    elif damage == "bad_json":
        marker.write_text("not json")
    else:
        marker.write_bytes(b" " * 65537)
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(tmp_path).verify_complete(identity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "other"),
        ("chunk_id", "other"),
        ("attempt_id", "other"),
        ("generation_fingerprint", "sha256:" + "b" * 64),
    ],
)
def test_marker_must_match_owned_attempt_and_fingerprint(tmp_path, field, value):
    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    identity, _, marker, record = _fixture(tmp_path)
    record["identity"][field] = value
    marker.write_text(json.dumps(record))
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(tmp_path).verify_complete(identity)


@pytest.mark.parametrize("target", ["audio", "marker", "directory"])
def test_symlink_escape_is_rejected(tmp_path, target):
    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    root = tmp_path / "managed"
    identity, audio, marker, _ = _fixture(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "directory":
        audio.rename(outside / audio.name)
        marker.rename(outside / marker.name)
        audio.parent.rmdir()
        audio.parent.symlink_to(outside, target_is_directory=True)
    else:
        path = audio if target == "audio" else marker
        external = outside / path.name
        path.rename(external)
        path.symlink_to(external)
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(root).verify_complete(identity)


@pytest.mark.parametrize(
    "unsafe", ["../escape", "/absolute", "..", "a/b", "a\\b", "C:drive", ""]
)
def test_identity_cannot_supply_filesystem_paths(unsafe):
    from pydantic import ValidationError

    from elevenlabs_mcp.artifacts import ArtifactIdentity

    for field in ("job_id", "chunk_id", "attempt_id"):
        values = {
            "job_id": "job",
            "chunk_id": "chunk",
            "attempt_id": "attempt",
            "generation_fingerprint": "sha256:" + "a" * 64,
        }
        values[field] = unsafe
        with pytest.raises(ValidationError):
            ArtifactIdentity.model_validate(values)


def test_nonregular_files_are_rejected_without_blocking(tmp_path):
    import os

    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    identity, audio, _, _ = _fixture(tmp_path)
    audio.unlink()
    os.mkfifo(audio)
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(tmp_path).verify_complete(identity)


def test_modified_source_during_read_is_rejected(tmp_path, monkeypatch):
    import os

    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    identity, audio, _, _ = _fixture(tmp_path)
    original = audio.read_bytes()
    read = os.read
    # Force multiple reads so the mutation overlaps unread bytes, regardless
    # of filesystem timestamp granularity.
    monkeypatch.setattr("elevenlabs_mcp.artifacts._READ_BLOCK", 16)

    def changing_read(fd, size):
        block = read(fd, size)
        if block == original[:16]:
            audio.write_bytes(b"x" * len(original))
        return block

    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(tmp_path).verify_complete(identity)


def test_source_size_policy_and_unsupported_platform_fail_explicitly(
    tmp_path, monkeypatch
):
    import os

    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    identity, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ArtifactVerificationError, match="size limit"):
        ArtifactVerifier(tmp_path, max_source_bytes=1).verify_complete(identity)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(ArtifactVerificationError, match="unavailable"):
        ArtifactVerifier(tmp_path).verify_complete(identity)


def test_unvalidated_model_copy_cannot_bypass_path_checks(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactVerificationError, ArtifactVerifier

    identity, _, _, _ = _fixture(tmp_path)
    forged = identity.model_copy(update={"job_id": "../outside"})
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(tmp_path).verify_complete(forged)
