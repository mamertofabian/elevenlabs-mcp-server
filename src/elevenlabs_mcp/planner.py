"""Pure deterministic planning primitives."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from typing import TypeAlias

import regex

from elevenlabs_mcp.contracts import (
    PlannedChunk,
    PlannedFragment,
    PlannedRequest,
    PlanningLimits,
    Script,
    ScriptPart,
    SourceSpan,
    VoiceoverOptions,
    VoiceoverPlan,
)

_TOKEN_PATTERN = regex.compile(r"\[[^\[\]]*\]|\X")
_PARAGRAPH_BOUNDARY = regex.compile(r"\n{2,}")
_SENTENCE_BOUNDARY = regex.compile(r"[.!?…。！？](?:[ \t]+|(?=\n|$))")
_WHITESPACE_BOUNDARY = regex.compile(r"\s+")


class TextFragmentationError(ValueError):
    def __init__(
        self, part_id: str, token_start: int, token_end: int, max_characters: int
    ) -> None:
        self.part_id = part_id
        self.token_start = token_start
        self.token_end = token_end
        self.max_characters = max_characters
        super().__init__(
            f"part {part_id!r} contains an indivisible token at "
            f"[{token_start}:{token_end}] exceeding limit {max_characters}"
        )


class PlanningLimitError(ValueError):
    def __init__(self, limit_name: str, limit: int, actual: int) -> None:
        self.limit_name = limit_name
        self.limit = limit
        self.actual = actual
        super().__init__(f"{limit_name} limit {limit} exceeded by {actual}")


@dataclass(frozen=True, slots=True)
class TextFragment:
    text: str
    source_span: SourceSpan

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text fragment must not be empty")
        if len(self.text) != self.source_span.end - self.source_span.start:
            raise ValueError("text fragment length must match its source span")


TextFragments: TypeAlias = tuple[TextFragment, ...]
PlannedChunks: TypeAlias = tuple[PlannedChunk, ...]


class ScriptPlanner:
    def plan(
        self, script: Script, options: VoiceoverOptions, limits: PlanningLimits
    ) -> VoiceoverPlan:
        normalized_script = _normalize_script(script)
        total_parts = sum(len(scene.parts) for scene in normalized_script.scenes)
        total_characters = sum(
            len(part.text) for scene in normalized_script.scenes for part in scene.parts
        )
        totals = (
            ("max_total_characters", limits.max_total_characters, total_characters),
            ("max_parts", limits.max_parts, total_parts),
            ("max_scenes", limits.max_scenes, len(normalized_script.scenes)),
            ("max_cast_entries", limits.max_cast_entries, len(normalized_script.cast)),
        )
        for limit_name, limit, actual in totals:
            if actual > limit:
                raise PlanningLimitError(limit_name, limit, actual)

        planner_version = "1"
        identity_payload = {
            "effective_limits": limits.model_dump(mode="json"),
            "normalized_script": normalized_script.model_dump(mode="json"),
            "planner_version": planner_version,
            "resolved_options": options.model_dump(mode="json"),
        }
        plan_hash = _sha256_json(identity_payload)
        chunks = chunk_script(normalized_script, options, limits)
        hash_prefix = plan_hash.removeprefix("sha256:")[:16]
        requests = tuple(
            _planned_request(chunk, options, hash_prefix) for chunk in chunks
        )
        warnings = (
            ("Acting continuity across generation requests is not guaranteed.",)
            if len(requests) > 1
            else ()
        )
        return VoiceoverPlan(
            plan_hash=plan_hash,
            planner_version=planner_version,
            normalized_script=normalized_script,
            resolved_options=options,
            effective_limits=limits,
            requests=requests,
            total_parts=total_parts,
            total_characters=total_characters,
            distinct_voice_count=len(
                {voice.voice_id for voice in normalized_script.cast.values()}
            ),
            warnings=warnings,
        )


def fragment_text(part: ScriptPart, max_characters: int) -> TextFragments:
    """Normalize line endings and split a part at stable lossless boundaries."""
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")

    text = part.text.replace("\r\n", "\n").replace("\r", "\n")
    token_ends = tuple(match.end() for match in _TOKEN_PATTERN.finditer(text))
    legal_boundaries = frozenset(token_ends)
    paragraph_ends = _boundary_ends(_PARAGRAPH_BOUNDARY, text, legal_boundaries)
    sentence_ends = _boundary_ends(_SENTENCE_BOUNDARY, text, legal_boundaries)
    whitespace_ends = _boundary_ends(_WHITESPACE_BOUNDARY, text, legal_boundaries)

    fragments: list[TextFragment] = []
    start = 0
    token_index = 0
    while start < len(text):
        token_end = token_ends[token_index]
        if token_end - start > max_characters:
            raise TextFragmentationError(part.id, start, token_end, max_characters)

        hard_end = min(start + max_characters, len(text))
        legal_stop = bisect_right(token_ends, hard_end, lo=token_index)
        end = (
            len(text)
            if hard_end == len(text)
            else _preferred_end(
                start,
                token_ends[legal_stop - 1],
                paragraph_ends,
                sentence_ends,
                whitespace_ends,
            )
        )
        fragments.append(
            TextFragment(
                text=text[start:end],
                source_span=SourceSpan(part_id=part.id, start=start, end=end),
            )
        )
        start = end
        token_index = bisect_right(token_ends, end, lo=token_index, hi=legal_stop)

    return tuple(fragments)


def chunk_script(
    script: Script, options: VoiceoverOptions, limits: PlanningLimits
) -> PlannedChunks:
    """Group normalized fragments into deterministic provider request chunks."""
    chunks: list[PlannedChunk] = []
    pending: list[PlannedFragment] = []
    pending_scene = ""
    pending_characters = 0
    pending_voice_ids: dict[str, None] = {}

    def flush(pause_after_ms: int = 0) -> None:
        nonlocal pending, pending_characters, pending_voice_ids
        if not pending:
            return
        actual = len(chunks) + 1
        if actual > limits.max_planned_chunks:
            raise PlanningLimitError(
                "max_planned_chunks", limits.max_planned_chunks, actual
            )
        chunks.append(
            PlannedChunk(
                index=len(chunks),
                scene_id=pending_scene,
                fragments=tuple(pending),
                character_count=pending_characters,
                voice_ids=tuple(pending_voice_ids),
                pause_after_ms=pause_after_ms,
            )
        )
        pending = []
        pending_characters = 0
        pending_voice_ids = {}

    for scene in script.scenes:
        flush()
        pending_scene = scene.id
        for part in scene.parts:
            voice_id = script.cast[part.actor].voice_id
            text_fragments = fragment_text(part, limits.max_text_characters)
            for fragment_index, fragment in enumerate(text_fragments):
                planned = PlannedFragment(
                    text=fragment.text,
                    actor=part.actor,
                    voice_id=voice_id,
                    source_span=fragment.source_span,
                )
                is_final_fragment = fragment_index == len(text_fragments) - 1
                pause_after_ms = part.pause_after_ms if is_final_fragment else 0

                if options.engine == "tts":
                    pending = [planned]
                    pending_characters = len(planned.text)
                    pending_voice_ids = {planned.voice_id: None}
                    flush(pause_after_ms)
                    continue

                exceeds_characters = (
                    pending_characters + len(planned.text) > limits.max_text_characters
                )
                exceeds_voices = (
                    planned.voice_id not in pending_voice_ids
                    and len(pending_voice_ids) >= limits.max_unique_voices
                )
                if pending and (exceeds_characters or exceeds_voices):
                    flush()
                pending.append(planned)
                pending_characters += len(planned.text)
                pending_voice_ids.setdefault(planned.voice_id, None)
                if pause_after_ms:
                    flush(pause_after_ms)
        flush()

    return tuple(chunks)


def _normalize_script(script: Script) -> Script:
    data = script.model_dump(mode="json")
    for scene in data["scenes"]:
        for part in scene["parts"]:
            part["text"] = part["text"].replace("\r\n", "\n").replace("\r", "\n")
    return Script.model_validate(data)


def _planned_request(
    chunk: PlannedChunk, options: VoiceoverOptions, hash_prefix: str
) -> PlannedRequest:
    seed = (
        None if options.seed is None else (options.seed + chunk.index) % 4_294_967_296
    )
    fingerprint_payload = {
        "chunk": chunk.model_dump(mode="json"),
        "engine": options.engine,
        "export_format": options.export_format,
        "language_code": options.language_code,
        "model_id": options.model_id,
        "seed": seed,
        "voice_settings": (
            None
            if options.voice_settings is None
            else options.voice_settings.model_dump(mode="json")
        ),
    }
    return PlannedRequest(
        chunk_id=f"chk_{hash_prefix}_{chunk.index:06d}",
        chunk=chunk,
        seed=seed,
        generation_fingerprint=_sha256_json(fingerprint_payload),
    )


def _sha256_json(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _boundary_ends(
    pattern: regex.Pattern[str], text: str, legal_boundaries: frozenset[int]
) -> tuple[int, ...]:
    return tuple(
        end
        for match in pattern.finditer(text)
        if (end := match.end()) in legal_boundaries
    )


def _preferred_end(
    start: int,
    hard_end: int,
    paragraph_ends: tuple[int, ...],
    sentence_ends: tuple[int, ...],
    whitespace_ends: tuple[int, ...],
) -> int:
    for preferred in (paragraph_ends, sentence_ends, whitespace_ends):
        index = bisect_right(preferred, hard_end) - 1
        if index >= 0 and preferred[index] > start:
            return preferred[index]
    return hard_end
