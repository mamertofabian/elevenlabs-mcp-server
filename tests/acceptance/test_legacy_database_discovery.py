from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import elevenlabs_mcp.server as server_module
from elevenlabs_mcp.config import Settings
from elevenlabs_mcp.server import ElevenLabsServer


def test_startup_refuses_fresh_database_when_different_legacy_history_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_cwd = tmp_path / "launch"
    legacy_database = launch_cwd / "output" / "voiceover_history.db"
    package_root = tmp_path / "package"
    package_database = package_root / "output" / "voiceover_history.db"
    legacy_database.parent.mkdir(parents=True)
    package_database.parent.mkdir(parents=True)
    legacy_database.write_bytes(b"synthetic legacy marker")
    package_database.write_bytes(b"synthetic package marker")
    requested_output = tmp_path / "configured" / "audio"
    requested_database = requested_output / "voiceover_history.db"
    settings = Settings(
        launch_cwd=launch_cwd,
        output_dir=requested_output,
        database_path=requested_database,
    )
    monkeypatch.setenv("ELEVENLABS_OUTPUT_DIR", str(requested_output))
    monkeypatch.delenv("ELEVENLABS_DATABASE_PATH", raising=False)
    fake_server_file = package_root / "src" / "elevenlabs_mcp" / "server.py"
    monkeypatch.setattr(server_module, "__file__", str(fake_server_file))
    server = ElevenLabsServer(settings, enable_revival=False)
    server.api.get_voices = list
    monkeypatch.setenv(
        "ELEVENLABS_DATABASE_PATH", str(tmp_path / "late-change" / "ignored.db")
    )

    with pytest.raises(RuntimeError) as captured:
        asyncio.run(server.initialize())

    from elevenlabs_mcp.config import DatabasePathAmbiguityError

    assert isinstance(captured.value, DatabasePathAmbiguityError)
    assert captured.value.existing_candidates == (
        legacy_database.resolve(),
        package_database.resolve(),
    )
    assert legacy_database.read_bytes() == b"synthetic legacy marker"
    assert package_database.read_bytes() == b"synthetic package marker"
    assert not requested_output.exists()
    assert not requested_database.exists()


def test_explicit_database_override_resolves_ambiguity(tmp_path: Path) -> None:
    from elevenlabs_mcp.config import select_database_path, settings_from_environment

    launch_cwd = tmp_path / "launch"
    package_root = tmp_path / "package"
    explicit_database = tmp_path / "selected" / "history.db"
    launch_legacy = launch_cwd / "output" / "voiceover_history.db"
    package_legacy = package_root / "output" / "voiceover_history.db"
    launch_legacy.parent.mkdir(parents=True)
    package_legacy.parent.mkdir(parents=True)
    launch_legacy.write_bytes(b"launch history")
    package_legacy.write_bytes(b"package history")
    environ = {"ELEVENLABS_DATABASE_PATH": str(explicit_database)}
    settings = settings_from_environment(environ, launch_cwd)

    selected = select_database_path(settings, package_root)

    assert selected == explicit_database.resolve()
    assert settings.database_path_explicit is True
    assert not explicit_database.exists()
    assert launch_legacy.read_bytes() == b"launch history"
    assert package_legacy.read_bytes() == b"package history"


def test_requested_database_is_selected_without_filesystem_writes(
    tmp_path: Path,
) -> None:
    from elevenlabs_mcp.config import select_database_path, settings_from_environment

    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    package_root = tmp_path / "package"
    settings = settings_from_environment({}, launch_cwd)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    selected = select_database_path(settings, package_root)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert selected == settings.database_path
    assert settings.database_path_explicit is False
    assert after == before
    assert not settings.database_path.exists()


def test_ambiguity_reports_deduplicated_known_candidates(tmp_path: Path) -> None:
    from elevenlabs_mcp.config import (
        DatabasePathAmbiguityError,
        select_database_path,
        settings_from_environment,
    )

    launch_cwd = tmp_path / "launch"
    package_root = tmp_path / "package"
    requested_output = tmp_path / "configured"
    launch_legacy = launch_cwd / "output" / "voiceover_history.db"
    package_legacy = package_root / "output" / "voiceover_history.db"
    launch_legacy.parent.mkdir(parents=True)
    package_legacy.parent.mkdir(parents=True)
    launch_legacy.write_bytes(b"private launch contents")
    package_legacy.write_bytes(b"private package contents")
    environ = {"ELEVENLABS_OUTPUT_DIR": str(requested_output)}
    settings = settings_from_environment(environ, launch_cwd)

    with pytest.raises(DatabasePathAmbiguityError) as captured:
        select_database_path(settings, package_root)

    error = captured.value
    assert error.requested_path == settings.database_path
    assert error.existing_candidates == (
        launch_legacy.resolve(),
        package_legacy.resolve(),
    )
    assert "private launch contents" not in str(error)
    assert "private package contents" not in str(error)

    with pytest.raises(DatabasePathAmbiguityError) as duplicate_capture:
        select_database_path(settings, launch_cwd)

    assert duplicate_capture.value.existing_candidates == (launch_legacy.resolve(),)
