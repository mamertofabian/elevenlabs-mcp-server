from __future__ import annotations

from time import perf_counter

import pytest


def _script(parts: list[dict], cast: dict[str, dict], scene_id: str = "scene"):
    from elevenlabs_mcp.contracts import Script

    return Script.model_validate(
        {
            "script_version": "1",
            "cast": cast,
            "scenes": [{"id": scene_id, "parts": parts}],
        }
    )


def test_dialogue_starts_new_chunk_for_eleventh_distinct_voice() -> None:
    from elevenlabs_mcp.contracts import (
        PlannedChunk,
        PlannedFragment,
        PlanningLimits,
        VoiceoverOptions,
    )
    from elevenlabs_mcp.planner import PlannedChunks, chunk_script

    cast = {f"actor{i}": {"voice_id": f"voice{i}"} for i in range(11)}
    parts = [{"id": f"part{i}", "actor": f"actor{i}", "text": "x"} for i in range(11)]
    limits = PlanningLimits()
    chunks: PlannedChunks = chunk_script(
        _script(parts, cast),
        VoiceoverOptions(engine="dialogue", model_id="eleven_v3"),
        limits,
    )

    assert PlannedChunks == tuple[PlannedChunk, ...]
    assert limits.max_text_characters == 2_000
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert all(
        isinstance(fragment, PlannedFragment)
        for chunk in chunks
        for fragment in chunk.fragments
    )
    assert chunks[0].fragments[0].voice_id == "voice0"
    assert [chunk.voice_ids for chunk in chunks] == [
        tuple(f"voice{i}" for i in range(10)),
        ("voice10",),
    ]
    assert [chunk.character_count for chunk in chunks] == [10, 1]


def test_dialogue_counts_shared_voice_id_once() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, VoiceoverOptions
    from elevenlabs_mcp.planner import chunk_script

    chunks = chunk_script(
        _script(
            [
                {"id": "p1", "actor": "alice", "text": "Hello."},
                {"id": "p2", "actor": "bob", "text": "World."},
            ],
            {"alice": {"voice_id": "shared"}, "bob": {"voice_id": "shared"}},
        ),
        VoiceoverOptions(engine="dialogue", model_id="eleven_v3"),
        PlanningLimits(max_unique_voices=1),
    )

    assert len(chunks) == 1
    assert chunks[0].voice_ids == ("shared",)
    assert [fragment.actor for fragment in chunks[0].fragments] == ["alice", "bob"]
    assert "".join(fragment.text for fragment in chunks[0].fragments) == "Hello.World."


def test_dialogue_character_ceiling_splits_at_exact_boundary() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, VoiceoverOptions
    from elevenlabs_mcp.planner import chunk_script

    chunks = chunk_script(
        _script(
            [
                {"id": "p1", "actor": "actor", "text": "a" * 1_000},
                {"id": "p2", "actor": "actor", "text": "b" * 1_000},
                {"id": "p3", "actor": "actor", "text": "c"},
            ],
            {"actor": {"voice_id": "voice"}},
        ),
        VoiceoverOptions(engine="dialogue", model_id="eleven_v3"),
        PlanningLimits(),
    )

    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.character_count for chunk in chunks] == [2_000, 1]
    assert (
        "".join(fragment.text for chunk in chunks for fragment in chunk.fragments)
        == "a" * 1_000 + "b" * 1_000 + "c"
    )


def test_scenes_and_explicit_pauses_force_stable_boundaries() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
    from elevenlabs_mcp.planner import chunk_script

    script = Script.model_validate(
        {
            "script_version": "1",
            "cast": {"actor": {"voice_id": "voice"}},
            "scenes": [
                {
                    "id": "first",
                    "parts": [
                        {
                            "id": "p1",
                            "actor": "actor",
                            "text": "One",
                            "pause_after_ms": 250,
                        },
                        {"id": "p2", "actor": "actor", "text": "Two"},
                    ],
                },
                {
                    "id": "second",
                    "parts": [{"id": "p3", "actor": "actor", "text": "Three"}],
                },
            ],
        }
    )
    chunks = chunk_script(
        script,
        VoiceoverOptions(engine="dialogue", model_id="eleven_v3"),
        PlanningLimits(),
    )

    assert [(chunk.scene_id, chunk.pause_after_ms) for chunk in chunks] == [
        ("first", 250),
        ("first", 0),
        ("second", 0),
    ]
    assert [
        [item.source_span.part_id for item in chunk.fragments] for chunk in chunks
    ] == [["p1"], ["p2"], ["p3"]]


def test_tts_emits_one_chunk_per_lossless_fragment_without_actor_labels() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, VoiceoverOptions
    from elevenlabs_mcp.planner import chunk_script

    text = "x" * 2_001
    chunks = chunk_script(
        _script(
            [{"id": "part", "actor": "narrator", "text": text}],
            {"narrator": {"voice_id": "voice"}},
        ),
        VoiceoverOptions(engine="tts", model_id="eleven_multilingual_v2"),
        PlanningLimits(),
    )

    assert len(chunks) == 2
    assert all(len(chunk.fragments) == 1 for chunk in chunks)
    assert all(chunk.voice_ids == ("voice",) for chunk in chunks)
    assert "".join(chunk.fragments[0].text for chunk in chunks) == text
    assert "narrator" not in "".join(chunk.fragments[0].text for chunk in chunks)


def test_planned_chunk_limit_fails_with_typed_evidence() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, VoiceoverOptions
    from elevenlabs_mcp.planner import PlanningLimitError, chunk_script

    with pytest.raises(PlanningLimitError) as captured:
        chunk_script(
            _script(
                [
                    {"id": "p1", "actor": "actor", "text": "One"},
                    {"id": "p2", "actor": "actor", "text": "Two"},
                ],
                {"actor": {"voice_id": "voice"}},
            ),
            VoiceoverOptions(engine="tts", model_id="eleven_multilingual_v2"),
            PlanningLimits(max_planned_chunks=1),
        )

    assert captured.value.limit_name == "max_planned_chunks"
    assert captured.value.limit == 1
    assert captured.value.actual == 2


def test_maximum_part_count_groups_in_bounded_time() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
    from elevenlabs_mcp.planner import PlannedChunks, chunk_script

    script = Script.model_validate(
        {
            "script_version": "1",
            "cast": {"actor": {"voice_id": "voice"}},
            "scenes": [
                {
                    "id": "scene",
                    "parts": [
                        {"id": f"part{i}", "actor": "actor", "text": "x"}
                        for i in range(2_000)
                    ],
                }
            ],
        }
    )

    started = perf_counter()
    chunks: PlannedChunks = ()
    for _ in range(30):
        chunks = chunk_script(
            script,
            VoiceoverOptions(engine="dialogue", model_id="eleven_v3"),
            PlanningLimits(),
        )
    elapsed = perf_counter() - started

    assert len(chunks) == 1
    assert chunks[0].character_count == 2_000
    assert elapsed < 3.0
