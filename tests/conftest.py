"""Credential and network safety defaults for the test suite."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import requests

# Set deterministic placeholders before application modules are imported during
# test collection. python-dotenv does not override these values by default.
_TEST_ROOT = TemporaryDirectory(prefix="elevenlabs-tests-")
os.environ["ELEVENLABS_API_KEY"] = "test-not-a-real-key"
os.environ["ELEVENLABS_VOICE_ID"] = "test-voice-id"
os.environ["ELEVENLABS_OUTPUT_DIR"] = str(Path(_TEST_ROOT.name) / "output")


@pytest.fixture(autouse=True)
def isolate_test_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Use an isolated cwd and fail every external connection attempt."""

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("External network access is disabled in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.chdir(tmp_path)
    yield
