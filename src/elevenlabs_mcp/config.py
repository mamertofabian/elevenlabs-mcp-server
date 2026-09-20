"""Runtime configuration resolution without import-time side effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved filesystem settings for one server process."""

    launch_cwd: Path
    output_dir: Path
    database_path: Path


def settings_from_environment(environ: Mapping[str, str], launch_cwd: Path) -> Settings:
    """Resolve paths from an explicit environment mapping without performing I/O."""

    resolved_launch_cwd = launch_cwd.resolve()
    output_dir = _resolve_path(
        environ.get("ELEVENLABS_OUTPUT_DIR") or "output", resolved_launch_cwd
    )
    database_value = environ.get("ELEVENLABS_DATABASE_PATH")
    database_path = (
        _resolve_path(database_value, resolved_launch_cwd)
        if database_value
        else output_dir / "voiceover_history.db"
    )
    return Settings(
        launch_cwd=resolved_launch_cwd,
        output_dir=output_dir,
        database_path=database_path,
    )


def _resolve_path(value: str, launch_cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = launch_cwd / path
    return path.resolve()
