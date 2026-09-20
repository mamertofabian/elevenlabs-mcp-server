from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
from tenacity import RetryError

from elevenlabs_mcp.elevenlabs_api import ElevenLabsAPI, UpstreamOutcomeUnknownError


class _Response:
    content = b""
    headers: ClassVar[dict[str, str]] = {}

    def __init__(self, status_code: int, text: str = "provider failure") -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, Any]:
        return {"voices": []}


def _disable_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    for method in (ElevenLabsAPI.get_voices, ElevenLabsAPI.generate_audio_segment):
        retry_state = cast(Any, method).retry
        monkeypatch.setattr(retry_state, "sleep", lambda _: None)


def test_voice_metadata_uses_finite_timeout_and_one_401_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    calls: list[dict[str, Any]] = []
    _disable_retry_sleep(monkeypatch)

    def get(*args: object, **kwargs: Any) -> _Response:
        del args
        calls.append(kwargs)
        return _Response(401)

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.get", get)

    with pytest.raises((RuntimeError, RetryError)):
        api.get_voices()

    assert len(calls) == 1
    assert calls[0]["timeout"] == (5.0, 60.0)


def test_synthesis_uses_finite_timeout_and_one_422_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    calls: list[dict[str, Any]] = []
    _disable_retry_sleep(monkeypatch)

    def post(*args: object, **kwargs: Any) -> _Response:
        del args
        calls.append(kwargs)
        return _Response(422)

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)

    with pytest.raises((RuntimeError, RetryError)):
        api.generate_audio_segment("Fixture", "voice-a", debug_info=[])

    assert len(calls) == 1
    assert calls[0]["timeout"] == (5.0, 60.0)


def test_metadata_500_retries_but_synthesis_5xx_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ElevenLabsAPI({"ELEVENLABS_API_KEY": "fixture"})
    get_calls = 0
    _disable_retry_sleep(monkeypatch)

    def get(*args: object, **kwargs: Any) -> _Response:
        nonlocal get_calls
        del args, kwargs
        get_calls += 1
        return _Response(500)

    monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.get", get)

    with pytest.raises(RetryError):
        api.get_voices()
    assert get_calls == 3

    for status_code in (500, 502, 503, 504):
        post_calls = 0

        def post(
            *args: object, response_status: int = status_code, **kwargs: Any
        ) -> _Response:
            nonlocal post_calls
            del args, kwargs
            post_calls += 1
            return _Response(response_status)

        monkeypatch.setattr("elevenlabs_mcp.elevenlabs_api.requests.post", post)
        with pytest.raises(UpstreamOutcomeUnknownError) as captured:
            api.generate_audio_segment("Fixture", "voice-a", debug_info=[])

        assert post_calls == 1
        assert captured.value.cause_type == f"HTTP_{status_code}"
