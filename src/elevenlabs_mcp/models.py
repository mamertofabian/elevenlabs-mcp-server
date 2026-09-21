from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ScriptPart:
    text: str
    voice_id: str | None = None
    actor: str | None = None


@dataclass
class AudioJob:
    id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    script_parts: list[dict]
    output_file: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    total_parts: int = 1
    completed_parts: int = 0

    @classmethod
    def create(
        cls,
        id: str,
        status: str,
        script_parts: list[dict],
        total_parts: int = 1,
        clock: Callable[[], datetime] = utc_now,
    ) -> "AudioJob":
        timestamp = _as_utc(clock())
        return cls(
            id=id,
            status=status,
            script_parts=script_parts,
            created_at=timestamp,
            updated_at=timestamp,
            total_parts=total_parts,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "script_parts": self.script_parts,
            "output_file": self.output_file,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_parts": self.total_parts,
            "completed_parts": self.completed_parts,
        }

    @staticmethod
    def from_dict(data: dict) -> "AudioJob":
        created_at = _as_utc(
            datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"]
        )
        updated_at = _as_utc(
            datetime.fromisoformat(data["updated_at"])
            if isinstance(data["updated_at"], str)
            else data["updated_at"]
        )
        return AudioJob(
            id=data["id"],
            status=data["status"],
            script_parts=data["script_parts"],
            output_file=data.get("output_file"),
            error=data.get("error"),
            created_at=created_at,
            updated_at=updated_at,
            total_parts=data.get("total_parts", 1),
            completed_parts=data.get("completed_parts", 0),
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
