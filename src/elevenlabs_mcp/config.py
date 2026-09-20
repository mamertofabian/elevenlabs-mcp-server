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
    database_path_explicit: bool = False


class DatabasePathAmbiguityError(RuntimeError):
    """Known legacy history exists outside the unresolved requested path."""

    def __init__(
        self, requested_path: Path, existing_candidates: tuple[Path, ...]
    ) -> None:
        self.requested_path = requested_path
        self.existing_candidates = existing_candidates
        candidates = ", ".join(str(path) for path in existing_candidates)
        super().__init__(
            "Existing legacy database history requires an explicit "
            f"ELEVENLABS_DATABASE_PATH selection. Requested: {requested_path}; "
            f"existing candidates: {candidates}"
        )


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
        database_path_explicit=bool(database_value),
    )


def select_database_path(
    settings: Settings,
    legacy_package_root: Path,
) -> Path:
    """Select the requested database or report different known legacy history."""

    requested_path = settings.database_path.resolve()
    if settings.database_path_explicit:
        return requested_path

    known_candidates = (
        settings.launch_cwd / "output" / "voiceover_history.db",
        legacy_package_root / "output" / "voiceover_history.db",
    )
    existing_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in known_candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen or resolved_candidate == requested_path:
            continue
        seen.add(resolved_candidate)
        if resolved_candidate.is_file():
            existing_candidates.append(resolved_candidate)

    if existing_candidates:
        raise DatabasePathAmbiguityError(
            requested_path=requested_path,
            existing_candidates=tuple(existing_candidates),
        )
    return requested_path


def _resolve_path(value: str, launch_cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = launch_cwd / path
    return path.resolve()
