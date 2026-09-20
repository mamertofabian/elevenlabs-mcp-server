from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from elevenlabs_mcp.server import ElevenLabsServer


def test_importing_api_and_server_never_calls_dotenv_loader(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    work_dir = sandbox / "work"
    home_dir = sandbox / "home"
    temp_dir = sandbox / "tmp"
    for path in (sandbox, work_dir, home_dir, temp_dir):
        path.mkdir()
    marker = sandbox / "dotenv-loader-called"
    command = (
        "import importlib, os, sys, types\n"
        "from pathlib import Path\n"
        "import dotenv\n"
        "def forbidden(*args, **kwargs):\n"
        " Path(os.environ['DOTENV_MARKER']).write_text('called', encoding='utf-8')\n"
        "dotenv.load_dotenv = forbidden\n"
        "package = types.ModuleType('elevenlabs_mcp')\n"
        "package.__path__ = [os.environ['PACKAGE_DIR']]\n"
        "sys.modules['elevenlabs_mcp'] = package\n"
        "importlib.import_module('elevenlabs_mcp.elevenlabs_api')\n"
        "importlib.import_module('elevenlabs_mcp.server')\n"
    )
    env = {
        **os.environ,
        "DOTENV_MARKER": str(marker),
        "ELEVENLABS_API_KEY": "",
        "HOME": str(home_dir),
        "TMPDIR": str(temp_dir),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PACKAGE_DIR": str(Path(__file__).parents[2] / "src" / "elevenlabs_mcp"),
    }

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_launch_dotenv_is_exact_and_operator_environment_wins(
    tmp_path: Path, monkeypatch: object
) -> None:
    from pytest import MonkeyPatch

    from elevenlabs_mcp.config import environment_from_dotenv

    assert isinstance(monkeypatch, MonkeyPatch)

    launch_cwd = tmp_path / "parent" / "launch"
    launch_cwd.mkdir(parents=True)
    (tmp_path / "parent" / ".env").write_text(
        "PARENT_ONLY=must-not-load\n", encoding="utf-8"
    )
    (launch_cwd / ".env").write_text(
        "ELEVENLABS_API_KEY=dotenv-key\n"
        "ELEVENLABS_OUTPUT_DIR=dotenv-output\n"
        "AMBIENT_REFERENCE=${AMBIENT_ONLY}\n"
        "EMPTY_VALUE=\n",
        encoding="utf-8",
    )
    operator = {
        "ELEVENLABS_API_KEY": "operator-key",
        "OPERATOR_ONLY": "present",
    }
    monkeypatch.setenv("AMBIENT_ONLY", "must-not-interpolate")

    merged = environment_from_dotenv(operator, launch_cwd)

    assert merged["ELEVENLABS_API_KEY"] == "operator-key"
    assert merged["ELEVENLABS_OUTPUT_DIR"] == "dotenv-output"
    assert merged["EMPTY_VALUE"] == ""
    assert merged["AMBIENT_REFERENCE"] == "${AMBIENT_ONLY}"
    assert merged["OPERATOR_ONLY"] == "present"
    assert "AMBIENT_ONLY" not in merged
    assert "PARENT_ONLY" not in merged


def test_environment_merge_does_not_mutate_inputs_or_process(tmp_path: Path) -> None:
    from elevenlabs_mcp.config import environment_from_dotenv

    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    (launch_cwd / ".env").write_text("SYNTHETIC_VALUE=dotenv\n", encoding="utf-8")
    supplied = {"SYNTHETIC_VALUE": "operator"}
    supplied_before = dict(supplied)
    process_before = dict(os.environ)

    merged = environment_from_dotenv(supplied, launch_cwd)
    merged["NEW_VALUE"] = "local-only"

    assert supplied == supplied_before
    assert dict(os.environ) == process_before
    assert "NEW_VALUE" not in supplied


def test_server_composition_passes_one_resolved_mapping(
    tmp_path: Path, monkeypatch: object
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    (launch_cwd / ".env").write_text(
        "ELEVENLABS_API_KEY=dotenv-key\nELEVENLABS_OUTPUT_DIR=dotenv-output\n",
        encoding="utf-8",
    )
    explicit_output = tmp_path / "operator-output"
    operator = {
        "ELEVENLABS_API_KEY": "operator-key",
        "ELEVENLABS_OUTPUT_DIR": str(explicit_output),
    }
    monkeypatch.chdir(launch_cwd)
    process_key_before = os.environ.get("ELEVENLABS_API_KEY")

    server = ElevenLabsServer(environ=operator)

    assert server.api.api_key == "operator-key"
    assert server.settings.output_dir == explicit_output.resolve()
    assert server.output_dir == explicit_output.resolve()
    assert os.environ.get("ELEVENLABS_API_KEY") == process_key_before
