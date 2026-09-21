"""Process-lifetime exclusive ownership for supported local workspaces."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import TracebackType
from typing import Self

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]


class WorkspaceBusyError(RuntimeError):
    """Another process owns the workspace."""


class WorkspaceOwnershipError(RuntimeError):
    """Exclusive ownership is absent or unavailable on this platform."""


class WorkspaceLock:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._descriptor: int | None = None
        self._pid: int | None = None
        self._directory_identity: tuple[int, int] | None = None

    def __enter__(self) -> Self:
        if self._descriptor is not None:
            raise WorkspaceOwnershipError(
                "Workspace lock is already held by this object"
            )
        if (
            fcntl is None
            or not hasattr(os, "O_NOFOLLOW")
            or os.open not in os.supports_dir_fd
        ):
            raise WorkspaceOwnershipError("Secure workspace locking is unavailable")
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(directory)
            directory_identity = (info.st_dev, info.st_ino)
            descriptor = os.open(
                ".elevenlabs-workspace.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise WorkspaceOwnershipError("Workspace lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkspaceBusyError("Workspace is already owned") from None
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self._pid = os.getpid()
        self._directory_identity = directory_identity
        return self

    def require_held(self, root: Path | None = None) -> None:
        if self._descriptor is None or self._pid != os.getpid():
            raise WorkspaceOwnershipError(
                "Current process must hold workspace ownership"
            )
        if root is not None:
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(directory)
                if (info.st_dev, info.st_ino) != self._directory_identity:
                    raise WorkspaceOwnershipError(
                        "Lock does not own this database directory"
                    )
            finally:
                os.close(directory)

    def require_database(self, path: Path) -> tuple[int, int]:
        """Bind recovery to a regular, nonaliased database under the held lock."""
        self.require_held(path.parent)
        try:
            directory = os.open(
                path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=directory,
                )
                try:
                    info = os.fstat(descriptor)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise WorkspaceOwnershipError(
                            "Recovery requires a nonaliased database file"
                        )
                    return info.st_dev, info.st_ino
                finally:
                    os.close(descriptor)
            finally:
                os.close(directory)
        except OSError:
            raise WorkspaceOwnershipError(
                "Recovery requires a nonaliased database file"
            ) from None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._descriptor is not None:
            # Closing releases the OS lock, including after process termination.
            # Never unlink the persistent inode: that would allow split ownership.
            os.close(self._descriptor)
            self._descriptor = None
            self._pid = None
            self._directory_identity = None
