"""Real synthetic audio shared by revival integration tests."""

import subprocess

import pytest


@pytest.fixture
def audio():
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-f",
            "mp3",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
