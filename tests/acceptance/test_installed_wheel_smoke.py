from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).parents[2]
EXPECTED_TOOLS = [
    "generate_audio_simple",
    "generate_audio_script",
    "delete_job",
    "get_audio_file",
    "list_voices",
    "get_voiceover_history",
]


def _environment_executables(environment: Path) -> tuple[Path, Path]:
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    console = scripts / (
        "elevenlabs-mcp-server.exe" if os.name == "nt" else "elevenlabs-mcp-server"
    )
    return python, console


@pytest.mark.parametrize(
    ("profile", "mcp_requirement"),
    [("lower", "mcp==2.2.0"), ("current-v2", None)],
)
def test_installed_wheel_stdio_matrix(
    tmp_path: Path, profile: str, mcp_requirement: str | None
) -> None:
    dist = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(dist.glob("elevenlabs_mcp_server-*.whl"))
    environment = tmp_path / f"venv-{profile}"
    create = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert create.returncode == 0, create.stderr
    python, console = _environment_executables(environment)
    install_command = ["uv", "pip", "install", "--python", str(python), str(wheel)]
    if mcp_requirement is not None:
        install_command.append(mcp_requirement)
    install = subprocess.run(
        install_command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert install.returncode == 0, install.stderr
    inspect_code = """
import importlib.metadata
import importlib.util
import json
import elevenlabs_mcp
import mcp
print(json.dumps({
    "distribution": importlib.metadata.version("elevenlabs-mcp-server"),
    "package": elevenlabs_mcp.__version__,
    "mcp": importlib.metadata.version("mcp"),
    "pytest": importlib.util.find_spec("pytest") is not None,
    "package_file": elevenlabs_mcp.__file__,
}))
"""
    inspected = subprocess.run(
        [str(python), "-c", inspect_code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert inspected.returncode == 0, inspected.stderr
    metadata = json.loads(inspected.stdout)
    assert metadata["distribution"] == "0.2.0.dev0"
    assert metadata["package"] == metadata["distribution"]
    assert metadata["pytest"] is False
    assert str(PROJECT_ROOT) not in metadata["package_file"]
    if profile == "lower":
        assert metadata["mcp"] == "2.2.0"
    else:
        assert int(metadata["mcp"].split(".", 1)[0]) == 2

    env = {
        **os.environ,
        "ELEVENLABS_API_KEY": "",
        "ELEVENLABS_OUTPUT_DIR": str(tmp_path / f"output-{profile}"),
        "ELEVENLABS_DATABASE_PATH": str(tmp_path / f"state-{profile}" / "history.db"),
        "ELEVENLABS_LOG_LEVEL": "ERROR",
        "PATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    parameters = StdioServerParameters(command=str(console), env=env)

    async def scenario() -> None:
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            async with asyncio.timeout(15):
                initialized = await session.initialize()
                tools = await session.list_tools()
                assert initialized.server_info.name == "elevenlabs-server"
                assert initialized.server_info.version == "0.2.0.dev0"
                assert [tool.name for tool in tools.tools][:6] == EXPECTED_TOOLS

    asyncio.run(scenario())
