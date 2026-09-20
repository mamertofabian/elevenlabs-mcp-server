from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest

from elevenlabs_mcp.database import Database


def test_importing_config_and_database_has_no_resource_side_effects(
    tmp_path: Path,
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    work_dir = sandbox / "work"
    home_dir = sandbox / "home"
    temp_dir = sandbox / "tmp"
    xdg_cache = sandbox / "xdg-cache"
    xdg_config = sandbox / "xdg-config"
    xdg_data = sandbox / "xdg-data"
    xdg_state = sandbox / "xdg-state"
    for path in (
        work_dir,
        home_dir,
        temp_dir,
        xdg_cache,
        xdg_config,
        xdg_data,
        xdg_state,
    ):
        path.mkdir()
    output_dir = work_dir / "not-created" / "output"
    network_marker = sandbox / "network-attempted"
    sitecustomize = sandbox / "sitecustomize.py"
    sitecustomize.write_text(
        """import os
import socket
from pathlib import Path
marker = Path(os.environ["R01_NETWORK_MARKER"])
def blocked(*args, **kwargs):
    marker.write_text("attempted", encoding="utf-8")
    raise RuntimeError("External network access is disabled")
socket.create_connection = blocked
socket.getaddrinfo = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.socket.send = blocked
socket.socket.sendall = blocked
socket.socket.sendto = blocked
socket.socket.sendmsg = blocked
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "ELEVENLABS_API_KEY": "test-not-a-real-key",
        "ELEVENLABS_OUTPUT_DIR": str(output_dir),
        "ELEVENLABS_DATABASE_PATH": str(work_dir / "not-created" / "history.db"),
        "HOME": str(home_dir),
        "TMPDIR": str(temp_dir),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "XDG_CACHE_HOME": str(xdg_cache),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_DATA_HOME": str(xdg_data),
        "XDG_STATE_HOME": str(xdg_state),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(sandbox), os.environ.get("PYTHONPATH")])
        ),
        "PYTHONUSERBASE": str(home_dir / ".local"),
        "R01_NETWORK_MARKER": str(network_marker),
        "R01_PACKAGE_DIR": str(Path(__file__).parents[2] / "src" / "elevenlabs_mcp"),
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket; socket.socket().connect_ex(('127.0.0.1', 9))",
        ],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert probe.returncode != 0
    assert network_marker.read_text(encoding="utf-8") == "attempted"
    network_marker.unlink()
    before = sorted(path.relative_to(sandbox) for path in sandbox.rglob("*"))
    command = [
        sys.executable,
        "-c",
        (
            "import importlib, os, sys, types\n"
            "package = types.ModuleType('elevenlabs_mcp')\n"
            "package.__path__ = [os.environ['R01_PACKAGE_DIR']]\n"
            "sys.modules['elevenlabs_mcp'] = package\n"
            "importlib.import_module('elevenlabs_mcp.database')\n"
            "try:\n"
            " importlib.import_module('elevenlabs_mcp.config')\n"
            "except ModuleNotFoundError:\n"
            " pass\n"
        ),
    ]

    result = subprocess.run(
        command,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    after = sorted(path.relative_to(sandbox) for path in sandbox.rglob("*"))
    assert after == before
    assert not output_dir.exists()
    assert not network_marker.exists()


def test_output_environment_resolves_default_database_under_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from elevenlabs_mcp.config import settings_from_environment

    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    monkeypatch.setenv("ELEVENLABS_OUTPUT_DIR", str(tmp_path / "process-environment"))

    defaults = settings_from_environment({}, launch_cwd)
    relative = settings_from_environment(
        {"ELEVENLABS_OUTPUT_DIR": "nested/audio"}, launch_cwd
    )
    explicit_database = settings_from_environment(
        {
            "ELEVENLABS_OUTPUT_DIR": "ignored-for-database",
            "ELEVENLABS_DATABASE_PATH": "state/history.sqlite3",
        },
        launch_cwd,
    )
    absolute_root = tmp_path / "absolute-output"
    absolute = settings_from_environment(
        {"ELEVENLABS_OUTPUT_DIR": str(absolute_root)}, launch_cwd
    )

    assert defaults.launch_cwd == launch_cwd.resolve()
    assert defaults.output_dir == (launch_cwd / "output").resolve()
    assert (
        defaults.database_path
        == (launch_cwd / "output" / "voiceover_history.db").resolve()
    )
    assert relative.output_dir == (launch_cwd / "nested/audio").resolve()
    assert (
        relative.database_path
        == (launch_cwd / "nested/audio/voiceover_history.db").resolve()
    )
    assert (
        explicit_database.database_path
        == (launch_cwd / "state/history.sqlite3").resolve()
    )
    assert absolute.output_dir == absolute_root.resolve()
    assert absolute.database_path == (absolute_root / "voiceover_history.db").resolve()
    assert not relative.output_dir.exists()
    assert not explicit_database.database_path.parent.exists()

    def mutate_output_dir(value: object) -> None:
        field_name = "output_" + "dir"
        setattr(value, field_name, tmp_path / "replacement")

    with pytest.raises(FrozenInstanceError):
        mutate_output_dir(defaults)


def test_explicit_database_path_is_injected_after_database_module_import(
    tmp_path: Path,
) -> None:
    import elevenlabs_mcp.database as imported_database
    from elevenlabs_mcp.config import settings_from_environment

    settings = settings_from_environment(
        {"ELEVENLABS_DATABASE_PATH": "runtime/selected.db"}, tmp_path
    )
    database = imported_database.Database(settings.database_path)
    no_arg_constructor = cast(Callable[[], Database], imported_database.Database)

    with pytest.raises(TypeError):
        no_arg_constructor()
    assert isinstance(database, Database)
    assert database.db_path == os.fspath(settings.database_path)
    assert not settings.database_path.parent.exists()


def test_initialize_creates_only_resolved_output_and_database_parents(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.config import Settings
    from elevenlabs_mcp.server import ElevenLabsServer

    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=tmp_path / "nested" / "audio",
        database_path=tmp_path / "nested" / "state" / "history.db",
    )
    server = ElevenLabsServer(settings)
    server.api.get_voices = list

    assert server.output_dir == settings.output_dir
    assert server.db.db_path == os.fspath(settings.database_path)
    assert not settings.output_dir.exists()
    assert not settings.database_path.parent.exists()

    asyncio.run(server.initialize())

    assert settings.output_dir.is_dir()
    assert settings.database_path.is_file()
    assert not (tmp_path / "output").exists()
