from __future__ import annotations

import asyncio
import importlib.metadata
import os
import sys
import tomllib
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import elevenlabs_mcp

PROJECT_ROOT = Path(__file__).parents[2]
DISTRIBUTION_NAME = "elevenlabs-mcp-server"


def test_package_export_matches_distribution_metadata() -> None:
    installed = importlib.metadata.version(DISTRIBUTION_NAME)

    assert elevenlabs_mcp.__version__ == installed
    from elevenlabs_mcp.version import package_version

    assert package_version() == installed


def test_source_fallback_matches_current_distribution_version(
    monkeypatch,
) -> None:
    from elevenlabs_mcp.version import package_version

    def missing(name: str) -> str:
        assert name == DISTRIBUTION_NAME
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert package_version() == "0.1.1"
    assert package_version() == pyproject["project"]["version"]


def test_real_mcp_initialization_reports_distribution_version(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "ELEVENLABS_API_KEY": "",
        "ELEVENLABS_OUTPUT_DIR": str(tmp_path / "output"),
        "ELEVENLABS_DATABASE_PATH": str(tmp_path / "state" / "history.db"),
        "ELEVENLABS_LOG_LEVEL": "ERROR",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "elevenlabs_mcp.server"],
        env=env,
    )

    async def scenario() -> None:
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            async with asyncio.timeout(10):
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "elevenlabs-server"
                assert initialized.serverInfo.version == importlib.metadata.version(
                    DISTRIBUTION_NAME
                )

    asyncio.run(scenario())

    from elevenlabs_mcp.version import package_version

    assert package_version() == importlib.metadata.version(DISTRIBUTION_NAME)
