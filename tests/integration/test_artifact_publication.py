from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from elevenlabs_mcp.artifacts import (
    ArtifactIdentity,
    ArtifactVerificationError,
    ArtifactVerifier,
)


def _identity():
    return ArtifactIdentity(
        job_id="job",
        chunk_id="chunk",
        attempt_id="attempt",
        generation_fingerprint="sha256:" + "a" * 64,
    )


def test_published_stream_has_verifiable_marker_and_exact_bytes(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactPublisher

    result = ArtifactPublisher(tmp_path).publish(
        _identity(), iter([b"first", b"", b"second"])
    )
    assert (tmp_path / result.relative_path).read_bytes() == b"firstsecond"
    assert ArtifactVerifier(tmp_path).verify_complete(_identity()) == result
    assert not list(tmp_path.rglob("*.tmp"))


def test_stream_interruption_preserves_original_error_without_publication(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactPublisher
    from elevenlabs_mcp.elevenlabs_api import UpstreamOutcomeUnknownError

    error = UpstreamOutcomeUnknownError("synthesis", "ReadTimeout")

    def interrupted():
        yield b"partial"
        raise error

    with pytest.raises(UpstreamOutcomeUnknownError) as caught:
        ArtifactPublisher(tmp_path).publish(_identity(), interrupted())
    assert caught.value is error
    assert not list(tmp_path.rglob("*.mp3"))
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("chunks", [[], [b""], [b"too large"], ["not bytes"]])
def test_invalid_or_oversized_source_leaves_no_files(tmp_path, chunks):
    from elevenlabs_mcp.artifacts import ArtifactPublicationError, ArtifactPublisher

    with pytest.raises(ArtifactPublicationError):
        ArtifactPublisher(tmp_path, max_source_bytes=4).publish(_identity(), chunks)
    assert all(path.is_dir() for path in tmp_path.rglob("*"))


def test_existing_attempt_is_not_overwritten_or_consumed(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactPublicationError, ArtifactPublisher

    publisher = ArtifactPublisher(tmp_path)
    result = publisher.publish(_identity(), [b"original"])

    def must_not_consume():
        raise AssertionError("duplicate publication consumed stream")
        yield b"unreachable"

    with pytest.raises(ArtifactPublicationError):
        publisher.publish(_identity(), must_not_consume())
    assert (tmp_path / result.relative_path).read_bytes() == b"original"


def test_directory_symlink_escape_is_rejected_without_external_writes(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactPublisher

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "jobs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        ArtifactPublisher(root).publish(_identity(), [b"audio"])
    assert not list(outside.iterdir())


def test_failed_marker_publication_leaves_unadoptable_source(tmp_path, monkeypatch):
    import os

    from elevenlabs_mcp.artifacts import ArtifactPublisher

    link = os.link

    def fail_marker(source, target, **kwargs):
        if target.endswith(".complete.json"):
            raise OSError("injected marker failure")
        return link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", fail_marker)
    # Capability checks must refer to the original platform implementation.
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, fail_marker})
    with pytest.raises(OSError, match="injected"):
        ArtifactPublisher(tmp_path).publish(_identity(), [b"audio"])
    assert len(list(tmp_path.rglob("*.mp3"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ArtifactVerificationError):
        ArtifactVerifier(tmp_path).verify_complete(_identity())


def test_concurrent_publishers_cannot_mix_or_overwrite_attempts(tmp_path):
    from elevenlabs_mcp.artifacts import ArtifactPublicationError, ArtifactPublisher

    def publish(payload):
        try:
            return ArtifactPublisher(tmp_path).publish(_identity(), [payload])
        except ArtifactPublicationError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, [b"one", b"two"]))
    assert sum(isinstance(result, ArtifactPublicationError) for result in results) == 1
    result = ArtifactVerifier(tmp_path).verify_complete(_identity())
    assert (tmp_path / result.relative_path).read_bytes() in {b"one", b"two"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_cleanup_failure_does_not_hide_uncertain_provider_outcome(
    tmp_path, monkeypatch
):
    import os

    from elevenlabs_mcp.artifacts import ArtifactPublisher
    from elevenlabs_mcp.elevenlabs_api import UpstreamOutcomeUnknownError

    def interrupted():
        yield b"partial"
        raise UpstreamOutcomeUnknownError("synthesis", "ReadTimeout")

    def failed_unlink(*args, **kwargs):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(os, "unlink", failed_unlink)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, failed_unlink})
    with pytest.raises(UpstreamOutcomeUnknownError) as caught:
        ArtifactPublisher(tmp_path).publish(_identity(), interrupted())
    assert "cleanup failed" in caught.value.__notes__[0]
    assert not list(tmp_path.rglob("*.complete.json"))
