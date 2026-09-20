from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError


def _fixture() -> dict:
    return {
        "script": {
            "script_version": "1",
            "title": "The workshop demonstration",
            "cast": {
                "narrator": {"voice_id": "REPLACE_WITH_PERMITTED_VOICE_A"},
                "builder": {"voice_id": "REPLACE_WITH_PERMITTED_VOICE_B"},
            },
            "scenes": [
                {
                    "id": "scene_1",
                    "parts": [
                        {
                            "id": "p1",
                            "actor": "narrator",
                            "text": "The workshop was quiet.",
                        },
                        {
                            "id": "p2",
                            "actor": "builder",
                            "text": "[curious] Shall we begin?",
                            "pause_after_ms": 500,
                        },
                    ],
                },
                {
                    "id": "scene_2",
                    "parts": [
                        {
                            "id": "p3",
                            "actor": "narrator",
                            "text": "Together, they began the demonstration.",
                        }
                    ],
                },
            ],
        },
        "options": {
            "engine": "dialogue",
            "model_id": "eleven_v3",
            "export_format": "mp3",
            "seed": 42,
        },
    }


def _plan(data: dict | None = None):
    from elevenlabs_mcp.contracts import PlanningLimits, PlanVoiceoverInput
    from elevenlabs_mcp.planner import ScriptPlanner

    value = PlanVoiceoverInput.model_validate(data or _fixture())
    return ScriptPlanner().plan(value.script, value.options, PlanningLimits())


def test_identical_input_produces_independently_reproducible_plan_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, VoiceoverPlan
    from elevenlabs_mcp.planner import ScriptPlanner

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("planning attempted external I/O")

    value = _fixture()
    first = _plan(value)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "open", forbidden)
        scoped.setattr(Path, "open", forbidden)
        scoped.setattr(socket, "create_connection", forbidden)
        scoped.setattr(os, "getenv", forbidden)
        scoped.setattr(os, "putenv", forbidden)
        second: VoiceoverPlan = _plan(value)

    payload = {
        "effective_limits": PlanningLimits().model_dump(mode="json"),
        "normalized_script": first.normalized_script.model_dump(mode="json"),
        "planner_version": first.planner_version,
        "resolved_options": first.resolved_options.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected = "sha256:" + hashlib.sha256(canonical).hexdigest()

    assert isinstance(ScriptPlanner(), ScriptPlanner)
    assert isinstance(first, VoiceoverPlan)
    assert first == second
    assert first.plan_hash == expected


def test_line_ending_equivalence_and_other_text_changes_affect_identity() -> None:
    base = _fixture()
    base["script"]["scenes"][0]["parts"][0]["text"] = "line one\nline two"
    crlf = copy.deepcopy(base)
    crlf["script"]["scenes"][0]["parts"][0]["text"] = "line one\r\nline two"
    cr = copy.deepcopy(base)
    cr["script"]["scenes"][0]["parts"][0]["text"] = "line one\rline two"
    changed = copy.deepcopy(base)
    changed["script"]["scenes"][0]["parts"][0]["text"] += " "

    lf_plan = _plan(base)
    assert _plan(crlf).plan_hash == lf_plan.plan_hash
    assert _plan(cr).plan_hash == lf_plan.plan_hash
    assert _plan(changed).plan_hash != lf_plan.plan_hash
    assert lf_plan.normalized_script.scenes[0].parts[0].text == "line one\nline two"


def test_request_ids_fingerprints_and_seeds_are_stable_and_input_bound() -> None:
    from elevenlabs_mcp.contracts import (
        PlannedRequest,
        PlanningLimits,
        PlanVoiceoverInput,
    )
    from elevenlabs_mcp.planner import ScriptPlanner

    data = _fixture()
    data["options"]["seed"] = 4_294_967_295
    value = PlanVoiceoverInput.model_validate(data)
    planner = ScriptPlanner()
    plan = planner.plan(
        value.script, value.options, PlanningLimits(max_text_characters=20)
    )

    assert all(isinstance(request, PlannedRequest) for request in plan.requests)
    assert plan.requests[0].chunk.index == 0
    assert [request.chunk_id for request in plan.requests] == [
        f"chk_{plan.plan_hash.removeprefix('sha256:')[:16]}_{index:06d}"
        for index in range(len(plan.requests))
    ]
    assert [request.seed for request in plan.requests[:2]] == [4_294_967_295, 0]
    assert all(
        request.generation_fingerprint.startswith("sha256:")
        for request in plan.requests
    )
    assert plan == planner.plan(
        value.script, value.options, PlanningLimits(max_text_characters=20)
    )

    other_limit = planner.plan(
        value.script, value.options, PlanningLimits(max_text_characters=21)
    )
    other_options = value.options.model_copy(update={"export_format": "wav"})
    other_format = planner.plan(
        value.script, other_options, PlanningLimits(max_text_characters=20)
    )
    assert other_limit.plan_hash != plan.plan_hash
    assert other_format.plan_hash != plan.plan_hash
    assert (
        other_format.requests[0].generation_fingerprint
        != plan.requests[0].generation_fingerprint
    )


def test_plan_summaries_and_safe_preview_fields_are_complete() -> None:
    from elevenlabs_mcp.contracts import PlanningLimits

    plan = _plan()
    source = _fixture()
    expected_characters = sum(
        len(part["text"].replace("\r\n", "\n").replace("\r", "\n"))
        for scene in source["script"]["scenes"]
        for part in scene["parts"]
    )

    assert plan.schema_version == "1"
    assert plan.planner_version == "1"
    assert plan.effective_limits == PlanningLimits()
    assert plan.effective_limits.max_total_characters == 100_000
    assert plan.effective_limits.max_parts == 2_000
    assert plan.effective_limits.max_scenes == 256
    assert plan.effective_limits.max_cast_entries == 128
    assert plan.total_parts == 3
    assert plan.total_characters == expected_characters
    assert plan.distinct_voice_count == 2
    assert len(plan.requests) >= 1
    assert plan.provider_access_checked is False
    assert plan.cost_estimate is None
    assert all(warning and len(warning) <= 1_000 for warning in plan.warnings)
    dumped = plan.model_dump(mode="json")
    assert dumped["plan_hash"] == plan.plan_hash
    assert "credential" not in json.dumps(dumped).lower()
    with pytest.raises(ValidationError):
        plan.total_parts = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("limit_name", "limit", "actual"),
    [
        ("max_total_characters", 1, 87),
        ("max_parts", 1, 3),
        ("max_scenes", 1, 2),
        ("max_cast_entries", 1, 2),
    ],
)
def test_job_wide_resource_limits_fail_before_chunking_side_effects(
    limit_name: str, limit: int, actual: int
) -> None:
    from elevenlabs_mcp.contracts import PlanningLimits, PlanVoiceoverInput
    from elevenlabs_mcp.planner import PlanningLimitError, ScriptPlanner

    value = PlanVoiceoverInput.model_validate(_fixture())
    limits = PlanningLimits(**{limit_name: limit})
    with pytest.raises(PlanningLimitError) as captured:
        ScriptPlanner().plan(value.script, value.options, limits)

    assert captured.value.limit_name == limit_name
    assert captured.value.limit == limit
    assert captured.value.actual == actual
