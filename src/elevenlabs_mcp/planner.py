"""Pure deterministic planning primitives."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TypeAlias

import regex

from elevenlabs_mcp.contracts import ScriptPart, SourceSpan

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
        end = _preferred_end(
            start,
            token_ends[legal_stop - 1],
            paragraph_ends,
            sentence_ends,
            whitespace_ends,
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
