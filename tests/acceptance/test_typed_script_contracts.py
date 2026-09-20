from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parents[2]


def _plan_fixture() -> dict:
    return json.loads(
        (PROJECT_ROOT / "docs/revival/examples/plan_voiceover.input.json").read_text(
            encoding="utf-8"
        )
    )


def test_supplied_plan_fixture_validates_and_preserves_values() -> None:
    from elevenlabs_mcp.contracts import (
        CastVoice,
        PlanVoiceoverInput,
        Scene,
        Script,
        ScriptPart,
        TtsVoiceSettings,
        VoiceoverOptions,
    )

    source = _plan_fixture()
    value = PlanVoiceoverInput.model_validate(source)
    script: Script = value.script
    options: VoiceoverOptions = value.options
    scene: Scene = script.scenes[0]
    part: ScriptPart = scene.parts[0]
    cast_voice: CastVoice = script.cast[part.actor]

    assert script.script_version == "1"
    assert script.title == "The workshop demonstration"
    assert set(script.cast) == {"narrator", "builder"}
    assert cast_voice.voice_id == "REPLACE_WITH_PERMITTED_VOICE_A"
    assert len(script.scenes) == 2
    assert scene.id == "scene_1"
    assert len(scene.parts) == 2
    assert part.id == "p1"
    assert part.actor == "narrator"
    assert part.text == "The workshop was quiet."
    assert part.pause_after_ms == 0
    assert options.engine == "dialogue"
    assert options.model_id == "eleven_v3"
    assert options.export_format == "mp3"
    assert options.seed == 42
    assert options.language_code is None
    assert options.voice_settings is None
    assert TtsVoiceSettings().model_dump() == {
        "stability": None,
        "similarity_boost": None,
        "style": None,
        "use_speaker_boost": None,
        "speed": None,
    }
    assert value.model_dump(mode="json", exclude_none=True) == {
        **source,
        "options": {**source["options"], "export_format": "mp3"},
        "script": {
            **source["script"],
            "scenes": [
                {
                    **raw_scene,
                    "parts": [
                        {
                            **raw_part,
                            "pause_after_ms": raw_part.get("pause_after_ms", 0),
                        }
                        for raw_part in raw_scene["parts"]
                    ],
                }
                for raw_scene in source["script"]["scenes"]
            ],
        },
    }


@pytest.mark.parametrize(
    "mutation", ["unknown_actor", "duplicate_scene", "duplicate_part", "blank_text"]
)
def test_script_semantic_relationships_fail_before_planning(mutation: str) -> None:
    from elevenlabs_mcp.contracts import Script

    data = copy.deepcopy(_plan_fixture()["script"])
    if mutation == "unknown_actor":
        data["scenes"][0]["parts"][0]["actor"] = "missing"
    elif mutation == "duplicate_scene":
        data["scenes"][1]["id"] = data["scenes"][0]["id"]
    elif mutation == "duplicate_part":
        data["scenes"][1]["parts"][0]["id"] = data["scenes"][0]["parts"][0]["id"]
    else:
        data["scenes"][0]["parts"][0]["text"] = " \t\n "

    with pytest.raises(ValidationError):
        Script.model_validate(data)


def test_nested_extra_keys_and_resource_boundaries_are_rejected() -> None:
    from elevenlabs_mcp.contracts import (
        CastVoice,
        PlanVoiceoverInput,
        Scene,
        Script,
        ScriptPart,
        TtsVoiceSettings,
        VoiceoverOptions,
    )

    valid = _plan_fixture()["script"]
    for invalid in (
        {**valid, "cast": {}},
        {**valid, "scenes": []},
        {**valid, "cast": {"bad name": {"voice_id": "voice"}}},
    ):
        with pytest.raises(ValidationError):
            Script.model_validate(invalid)

    valid_part = {"id": "part", "actor": "actor", "text": " text "}
    valid_scene = {"id": "scene", "parts": [valid_part]}
    valid_script = {
        "script_version": "1",
        "cast": {"actor": {"voice_id": "voice"}},
        "scenes": [valid_scene],
    }
    valid_options = {"engine": "tts", "model_id": "model"}
    extra_cases = (
        (
            PlanVoiceoverInput,
            {"script": valid_script, "options": valid_options, "extra": True},
        ),
        (Script, {**valid_script, "extra": True}),
        (CastVoice, {"voice_id": "voice", "extra": True}),
        (Scene, {**valid_scene, "extra": True}),
        (ScriptPart, {**valid_part, "extra": True}),
        (VoiceoverOptions, {**valid_options, "extra": True}),
        (TtsVoiceSettings, {"speed": 1.0, "extra": True}),
    )
    for model, data in extra_cases:
        with pytest.raises(ValidationError):
            model.model_validate(data)

    assert CastVoice.model_validate({"voice_id": "v" * 128}).voice_id == "v" * 128
    with pytest.raises(ValidationError):
        CastVoice.model_validate({"voice_id": "v" * 129})

    for pause in (0, 10_000):
        part = ScriptPart.model_validate(
            {
                **valid_part,
                "id": "p" * 64,
                "text": " " + "x" * 99_998 + " ",
                "pause_after_ms": pause,
            }
        )
        assert len(part.text) == 100_000
        assert part.text.startswith(" ") and part.text.endswith(" ")
    for invalid_part in (
        {**valid_part, "id": "p" * 65},
        {**valid_part, "text": "x" * 100_001},
        {**valid_part, "pause_after_ms": -1},
        {**valid_part, "pause_after_ms": 10_001},
    ):
        with pytest.raises(ValidationError):
            ScriptPart.model_validate(invalid_part)

    max_parts = [
        {"id": f"p{i}", "actor": "actor", "text": "text"} for i in range(2_001)
    ]
    assert (
        len(Scene.model_validate({"id": "scene", "parts": max_parts[:2_000]}).parts)
        == 2_000
    )
    with pytest.raises(ValidationError):
        Scene.model_validate({"id": "scene", "parts": max_parts})

    max_cast = {f"actor{i}": {"voice_id": "voice"} for i in range(129)}
    max_scenes = [
        {
            "id": f"scene{i}",
            "parts": [{"id": f"part{i}", "actor": "actor0", "text": "text"}],
        }
        for i in range(257)
    ]
    boundary_script = {
        "script_version": "1",
        "title": "t" * 200,
        "cast": dict(list(max_cast.items())[:128]),
        "scenes": max_scenes[:256],
    }
    result = Script.model_validate(boundary_script)
    assert len(result.cast) == 128
    assert len(result.scenes) == 256
    assert len(result.title or "") == 200
    for invalid_script in (
        {**boundary_script, "title": "t" * 201},
        {**boundary_script, "cast": max_cast},
        {**boundary_script, "scenes": max_scenes},
    ):
        with pytest.raises(ValidationError):
            Script.model_validate(invalid_script)

    with pytest.raises(TypeError):
        result.cast["new_actor"] = CastVoice(voice_id="voice")  # type: ignore[index]
    with pytest.raises(TypeError):
        del result.cast["actor0"]  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        result.cast["actor0"].voice_id = "replacement"  # type: ignore[misc]
    assert len(result.cast) == 128
    assert result.cast["actor0"].voice_id == "voice"


def test_voiceover_options_enforce_engine_profiles_and_bounds() -> None:
    from elevenlabs_mcp.contracts import TtsVoiceSettings, VoiceoverOptions

    dialogue = VoiceoverOptions.model_validate(
        {"engine": "dialogue", "model_id": "eleven_v3"}
    )
    tts = VoiceoverOptions.model_validate(
        {
            "engine": "tts",
            "model_id": "eleven_multilingual_v2",
            "export_format": "wav",
            "seed": 4294967295,
            "language_code": "en-US",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.1,
                "use_speaker_boost": True,
                "speed": 1.2,
            },
        }
    )

    assert dialogue.export_format == "mp3"
    assert isinstance(tts.voice_settings, TtsVoiceSettings)
    assert tts.voice_settings.stability == 0.5
    assert tts.voice_settings.similarity_boost == 0.75
    assert tts.voice_settings.style == 0.1
    assert tts.voice_settings.use_speaker_boost is True
    assert tts.voice_settings.speed == 1.2
    assert (
        VoiceoverOptions.model_validate(
            {"engine": "tts", "model_id": "m" * 128}
        ).model_id
        == "m" * 128
    )

    invalid_values = [
        {"engine": "dialogue", "model_id": "wrong"},
        {
            "engine": "dialogue",
            "model_id": "eleven_v3",
            "voice_settings": {},
        },
        {"engine": "tts", "model_id": "model", "seed": 4294967296},
        {"engine": "tts", "model_id": "m" * 129},
        {"engine": "tts", "model_id": "model", "language_code": "EN_us"},
        {
            "engine": "tts",
            "model_id": "model",
            "voice_settings": {"speed": 1.21},
        },
        {"engine": "tts", "model_id": "model", "extra": True},
    ]
    for invalid in invalid_values:
        with pytest.raises(ValidationError):
            VoiceoverOptions.model_validate(invalid)
