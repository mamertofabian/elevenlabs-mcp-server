from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class ToolFixture(TypedDict):
    name: str
    input: dict[str, Any]
    result: dict[str, Any]


class ResourceFixture(TypedDict):
    history_uri: str
    history_mime_type_consumed_by_client: str
    voices_uri: str


class LegacyFixture(TypedDict):
    fixture_version: str
    tools: list[ToolFixture]
    resources: ResourceFixture


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "legacy" / "tool_contracts.json"


def load_fixture() -> LegacyFixture:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_preserves_all_six_legacy_tool_names() -> None:
    fixture = load_fixture()
    tools = fixture["tools"]

    assert [tool["name"] for tool in tools] == [
        "generate_audio_simple",
        "generate_audio_script",
        "delete_job",
        "get_audio_file",
        "list_voices",
        "get_voiceover_history",
    ]


def test_generation_fixtures_preserve_old_client_success_signal() -> None:
    fixture = load_fixture()
    tools = {tool["name"]: tool for tool in fixture["tools"]}

    for name in ("generate_audio_simple", "generate_audio_script"):
        result = tools[name]["result"]
        assert result["text_first_line"] == "Audio generation successful."
        assert "successful" in result["text_first_line"]
        assert result["embedded_resource_mime_type"] == "audio/mpeg"


def test_history_fixture_preserves_old_client_shape() -> None:
    fixture = load_fixture()
    resources = fixture["resources"]
    tools = {tool["name"]: tool for tool in fixture["tools"]}

    assert resources["history_uri"] == "voiceover://history"
    assert resources["history_mime_type_consumed_by_client"] == "text/plain"
    assert tools["get_voiceover_history"]["result"]["history_fields"] == [
        "id",
        "status",
        "script_parts",
        "output_file",
        "error",
        "created_at",
        "updated_at",
        "total_parts",
        "completed_parts",
    ]
