from __future__ import annotations

from typing import Any, ClassVar

import pytest

from elevenlabs_mcp.elevenlabs_api import ElevenLabsAPI


class _Response:
    status_code = 200
    content = b"audio"
    headers: ClassVar[dict[str, str]] = {"request-id": "request-a"}
    text = ""


def _capture_body(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> tuple[dict[str, Any], tuple[bytes, str | None]]:
    api = ElevenLabsAPI(
        {
            "ELEVENLABS_API_KEY": "fixture",
            "ELEVENLABS_MODEL_ID": model_id,
            "ELEVENLABS_STABILITY": "0.4",
            "ELEVENLABS_SIMILARITY_BOOST": "0.6",
            "ELEVENLABS_STYLE": "0.2",
        }
    )
    captured: dict[str, Any] = {}

    def post(*args: object, **kwargs: Any) -> _Response:
        del args
        captured.update(kwargs["json"])
        return _Response()

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
    result = api.generate_audio_segment("Fixture", "voice-a", debug_info=[])
    return captured, result


def test_supported_model_sends_style_only_inside_voice_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body, result = _capture_body(monkeypatch, "eleven_multilingual_v2")

    assert "style" not in body
    assert body["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.6,
        "style": 0.2,
    }
    assert result == (b"audio", "request-a")


def test_unsupported_model_omits_style_at_every_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body, result = _capture_body(monkeypatch, "eleven_flash_v2_5")

    assert "style" not in body
    assert "style" not in body["voice_settings"]
    assert body["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.6,
    }
    assert result == (b"audio", "request-a")
