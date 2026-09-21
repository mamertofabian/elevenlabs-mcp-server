"""Strict, side-effect-free public input contracts for voiceover planning."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
_IDENTIFIER_RE = re.compile(_IDENTIFIER_PATTERN)
_LANGUAGE_PATTERN = r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$"
_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)


class _StrictModel(BaseModel):
    model_config = _STRICT_CONFIG


class CastVoice(_StrictModel):
    voice_id: str = Field(min_length=1, max_length=128)


class ScriptPart(_StrictModel):
    id: str = Field(pattern=_IDENTIFIER_PATTERN)
    actor: str = Field(pattern=_IDENTIFIER_PATTERN)
    text: str = Field(min_length=1, max_length=100_000)
    pause_after_ms: int = Field(default=0, ge=0, le=10_000)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace content")
        return value


class Scene(_StrictModel):
    id: str = Field(pattern=_IDENTIFIER_PATTERN)
    parts: tuple[ScriptPart, ...] = Field(min_length=1, max_length=2_000)

    @field_validator("parts", mode="before")
    @classmethod
    def _freeze_parts(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class Script(_StrictModel):
    script_version: Literal["1"]
    title: str | None = Field(default=None, min_length=1, max_length=200)
    cast: Mapping[str, CastVoice] = Field(min_length=1, max_length=128)
    scenes: tuple[Scene, ...] = Field(min_length=1, max_length=256)

    @field_validator("scenes", mode="before")
    @classmethod
    def _freeze_scenes(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        invalid_cast_ids = [
            key for key in self.cast if not _IDENTIFIER_RE.fullmatch(key)
        ]
        if invalid_cast_ids:
            raise ValueError("cast keys must be valid identifiers")

        scene_ids: set[str] = set()
        part_ids: set[str] = set()
        for scene in self.scenes:
            if scene.id in scene_ids:
                raise ValueError(f"duplicate scene id: {scene.id}")
            scene_ids.add(scene.id)
            for part in scene.parts:
                if part.id in part_ids:
                    raise ValueError(f"duplicate part id: {part.id}")
                part_ids.add(part.id)
                if part.actor not in self.cast:
                    raise ValueError(f"unknown actor: {part.actor}")
        object.__setattr__(self, "cast", MappingProxyType(dict(self.cast)))
        return self

    @field_serializer("cast")
    def _serialize_cast(self, value: Mapping[str, CastVoice]) -> dict[str, CastVoice]:
        return dict(value)


class TtsVoiceSettings(_StrictModel):
    stability: float | None = Field(default=None, ge=0, le=1)
    similarity_boost: float | None = Field(default=None, ge=0, le=1)
    style: float | None = Field(default=None, ge=0, le=1)
    use_speaker_boost: bool | None = None
    speed: float | None = Field(default=None, ge=0.7, le=1.2)


class VoiceoverOptions(_StrictModel):
    engine: Literal["dialogue", "tts"]
    model_id: str = Field(min_length=1, max_length=128)
    export_format: Literal["mp3", "wav"] = "mp3"
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    language_code: str | None = Field(default=None, pattern=_LANGUAGE_PATTERN)
    voice_settings: TtsVoiceSettings | None = None

    @model_validator(mode="after")
    def _validate_engine_profile(self) -> Self:
        if self.engine == "dialogue":
            if self.model_id != "eleven_v3":
                raise ValueError("dialogue engine requires model_id eleven_v3")
            if self.voice_settings is not None:
                raise ValueError("dialogue engine does not accept voice_settings")
        return self


class SourceSpan(_StrictModel):
    part_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class PlanningLimits(_StrictModel):
    max_total_characters: int = Field(default=100_000, ge=1, le=100_000)
    max_parts: int = Field(default=2_000, ge=1, le=2_000)
    max_scenes: int = Field(default=256, ge=1, le=256)
    max_cast_entries: int = Field(default=128, ge=1, le=128)
    max_text_characters: int = Field(default=2_000, ge=1, le=2_000)
    max_unique_voices: int = Field(default=10, ge=1, le=10)
    max_planned_chunks: int = Field(default=2_048, ge=1, le=2_048)


class PlannedFragment(_StrictModel):
    text: str = Field(min_length=1, max_length=2_000)
    actor: str = Field(pattern=_IDENTIFIER_PATTERN)
    voice_id: str = Field(min_length=1, max_length=128)
    source_span: SourceSpan

    @model_validator(mode="after")
    def _validate_span_length(self) -> Self:
        if len(self.text) != self.source_span.end - self.source_span.start:
            raise ValueError("planned fragment length must match its source span")
        return self


class PlannedChunk(_StrictModel):
    index: int = Field(ge=0)
    scene_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    fragments: tuple[PlannedFragment, ...] = Field(min_length=1, max_length=2_000)
    character_count: int = Field(ge=1, le=2_000)
    voice_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    pause_after_ms: int = Field(default=0, ge=0, le=10_000)

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        if self.character_count != sum(len(item.text) for item in self.fragments):
            raise ValueError("character_count must equal fragment text length")
        resolved = tuple(dict.fromkeys(item.voice_id for item in self.fragments))
        if self.voice_ids != resolved:
            raise ValueError("voice_ids must be stable unique fragment voice IDs")
        return self


class PlannedRequest(_StrictModel):
    chunk_id: str = Field(pattern=r"^chk_[0-9a-f]{16}_[0-9]{6}$")
    chunk: PlannedChunk
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    generation_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class VoiceoverPlan(_StrictModel):
    schema_version: Literal["1"] = "1"
    plan_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    planner_version: str = Field(min_length=1, max_length=64)
    normalized_script: Script
    resolved_options: VoiceoverOptions
    effective_limits: PlanningLimits
    requests: tuple[PlannedRequest, ...] = Field(min_length=1, max_length=2_048)
    total_parts: int = Field(ge=1, le=2_000)
    total_characters: int = Field(ge=1, le=100_000)
    distinct_voice_count: int = Field(ge=1, le=128)
    warnings: tuple[str, ...] = Field(max_length=100)
    provider_access_checked: Literal[False] = False
    cost_estimate: None = None


class JobCreateResult(_StrictModel):
    job_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    created: bool
    idempotent_replay: bool


class AttemptReservation(_StrictModel):
    attempt_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    reserved_characters: int = Field(ge=1)
    reserved_requests: int = Field(default=1, ge=1)
    replayed: bool


class PlanVoiceoverInput(_StrictModel):
    script: Script
    options: VoiceoverOptions


class AttemptDispatch(_StrictModel):
    """A newly committed dispatch; this result must never be replayed as permission."""

    attempt_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)


class AttemptFailureResult(_StrictModel):
    attempt_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    outcome: Literal["failed", "unknown"]
    replayed: bool


class CancellationResult(_StrictModel):
    """Original acknowledgment; a request flag does not mean upstream stopped."""

    job_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    status: Literal[
        "queued", "running", "assembling", "completed", "failed", "paused", "cancelled"
    ]
    reason: str | None
    cancel_requested: bool
    replayed: bool


class ResumeResult(_StrictModel):
    job_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    status: Literal["queued", "completed"]
    warnings: tuple[str, ...] = ()
    replayed: bool
