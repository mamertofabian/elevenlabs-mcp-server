from __future__ import annotations

import builtins
import importlib
import os
import socket
import tomllib
from pathlib import Path

import pytest


def _part(text: str):
    from elevenlabs_mcp.contracts import ScriptPart

    return ScriptPart(id="part_1", actor="narrator", text=text)


def test_exact_character_boundary_and_overflow_are_lossless() -> None:
    from elevenlabs_mcp.planner import TextFragment, TextFragments, fragment_text

    from elevenlabs_mcp.contracts import SourceSpan

    exact: TextFragments = fragment_text(_part("x" * 2_000), 2_000)
    overflow = fragment_text(_part("x" * 2_001), 2_000)

    assert TextFragments == tuple[TextFragment, ...]
    assert exact == (
        TextFragment(
            text="x" * 2_000,
            source_span=SourceSpan(part_id="part_1", start=0, end=2_000),
        ),
    )
    assert all(fragment.text for fragment in overflow)
    assert all(len(fragment.text) <= 2_000 for fragment in overflow)
    assert "".join(fragment.text for fragment in overflow) == "x" * 2_001
    assert [(item.source_span.start, item.source_span.end) for item in overflow] == [
        (0, 2_000),
        (2_000, 2_001),
    ]


def test_normalization_preserves_all_content_except_line_endings() -> None:
    from elevenlabs_mcp.planner import fragment_text

    source = "  First\r\n\r\n[curious] café 👩🏽‍🔧!\rTrailing spaces  "
    expected = "  First\n\n[curious] café 👩🏽‍🔧!\nTrailing spaces  "
    fragments = fragment_text(_part(source), 18)

    assert "".join(fragment.text for fragment in fragments) == expected
    assert fragments[0].source_span.start == 0
    assert fragments[-1].source_span.end == len(expected)
    assert all(
        expected[item.source_span.start : item.source_span.end] == item.text
        for item in fragments
    )


def test_maximum_text_with_unit_ceiling_completes_losslessly() -> None:
    from elevenlabs_mcp.planner import fragment_text

    source = "x" * 100_000
    fragments = fragment_text(_part(source), 1)

    assert len(fragments) == 100_000
    assert "".join(fragment.text for fragment in fragments) == source
    assert fragments[0].source_span.start == 0
    assert fragments[-1].source_span.end == 100_000


def test_fragmentation_respects_preferred_and_indivisible_boundaries() -> None:
    from elevenlabs_mcp.planner import fragment_text

    paragraph = fragment_text(_part("Alpha beta.\n\nGamma delta."), 15)
    sentence = fragment_text(_part("Alpha beta. Gamma delta."), 15)
    whitespace = fragment_text(_part("abcdefgh ijklmnop"), 12)
    indivisible = fragment_text(_part("a" * 8 + "[curious]" + "e\u0301" + "👩🏽‍🔧"), 10)

    assert [item.text for item in paragraph] == ["Alpha beta.\n\n", "Gamma delta."]
    assert [item.text for item in sentence] == ["Alpha beta. ", "Gamma delta."]
    assert [item.text for item in whitespace] == ["abcdefgh ", "ijklmnop"]
    assert (
        "".join(item.text for item in indivisible)
        == "a" * 8 + "[curious]" + "e\u0301" + "👩🏽‍🔧"
    )
    for token in ("[curious]", "e\u0301", "👩🏽‍🔧"):
        assert any(token in item.text for item in indivisible)


def test_indivisible_token_reports_precise_failure_and_planner_has_no_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elevenlabs_mcp.planner import TextFragmentationError, fragment_text

    with pytest.raises(TextFragmentationError) as captured:
        fragment_text(_part("before[" + "x" * 20 + "]after"), 10)
    assert captured.value.part_id == "part_1"
    assert captured.value.token_start == 6
    assert captured.value.token_end == 28
    assert captured.value.max_characters == 10

    with pytest.raises(ValueError):
        fragment_text(_part("text"), 0)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fragmentation attempted external I/O")

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "open", forbidden)
        scoped.setattr(Path, "open", forbidden)
        scoped.setattr(socket, "create_connection", forbidden)
        scoped.setattr(os, "getenv", forbidden)
        scoped.setattr(os, "putenv", forbidden)
        assert (
            "".join(item.text for item in fragment_text(_part("offline planning"), 8))
            == "offline planning"
        )


def test_unicode_segmenter_is_a_direct_runtime_dependency() -> None:
    project_root = Path(__file__).parents[2]
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert "regex>=2024.11.6,<2027" in project["dependencies"]
    assert importlib.import_module("regex").fullmatch(r"\X", "👩🏽‍🔧") is not None
