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


class PlanVoiceoverInput(_StrictModel):
    script: Script
    options: VoiceoverOptions
