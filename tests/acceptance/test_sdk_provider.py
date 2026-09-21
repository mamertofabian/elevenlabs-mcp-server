from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest

from elevenlabs_mcp.contracts import PlanningLimits, Script, VoiceoverOptions
from elevenlabs_mcp.planner import ScriptPlanner
from elevenlabs_mcp.provider import ElevenLabsProvider, ProviderError


def plan(engine: Literal["tts", "dialogue"] = "tts"):
    script = Script.model_validate(
        {
            "script_version": "1",
            "cast": {"a": {"voice_id": "voice-a"}, "b": {"voice_id": "voice-b"}},
            "scenes": [
                {
                    "id": "s",
                    "parts": [
                        {"id": "p", "actor": "a", "text": "First."},
                        {"id": "q", "actor": "b", "text": "Second."},
                    ],
                }
            ],
        }
    )
    options = VoiceoverOptions(
        engine=engine,
        model_id="eleven_v3" if engine == "dialogue" else "eleven_multilingual_v2",
        seed=7,
    )
    return ScriptPlanner().plan(script, options, PlanningLimits())


@pytest.mark.parametrize("engine", ["tts", "dialogue"])
def test_official_sdk_sends_expected_mode_payload_without_hidden_retries(engine):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, content=b"audio")

    provider = ElevenLabsProvider(
        "fixture", httpx.Client(transport=httpx.MockTransport(handle))
    )
    p = plan(engine)
    assert b"".join(provider.generate(p.requests[0], p.resolved_options)) == b"audio"
    assert len(calls) == 1
    body = json.loads(calls[0].content)
    assert body["seed"] == 7
    assert calls[0].url.params["output_format"] == "mp3_44100_128"
    if engine == "tts":
        assert body["text"] == "First." and "/text-to-speech/voice-a" in str(
            calls[0].url
        )
    else:
        assert body["inputs"] == [
            {"text": "First.", "voice_id": "voice-a"},
            {"text": "Second.", "voice_id": "voice-b"},
        ]


@pytest.mark.parametrize(
    "status,uncertain",
    [(401, False), (422, False), (429, False), (500, True), (503, True)],
)
def test_sdk_rejections_have_one_wire_attempt_and_sanitized_errors(status, uncertain):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"detail": "PRIVATE_SENTINEL"})

    provider = ElevenLabsProvider(
        "fixture", httpx.Client(transport=httpx.MockTransport(handle))
    )
    p = plan()
    with pytest.raises(ProviderError) as caught:
        list(provider.generate(p.requests[0], p.resolved_options))
    assert caught.value.uncertain is uncertain
    assert "PRIVATE" not in str(caught.value)
    assert len(calls) == 1


def test_sdk_interrupted_response_is_uncertain_and_never_replayed():
    calls = []

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("PRIVATE_SENTINEL")

    def handle(request):
        calls.append(request)
        return httpx.Response(200, stream=Interrupted())

    provider = ElevenLabsProvider(
        "fixture", httpx.Client(transport=httpx.MockTransport(handle))
    )
    p = plan()
    with pytest.raises(ProviderError) as caught:
        list(provider.generate(p.requests[0], p.resolved_options))
    assert caught.value.uncertain and len(calls) == 1
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize(
    "voice_id",
    [
        "../voices/add",
        "voice?output_format=wav",
        "%2e%2e%2fvoices",
        "voice/other",
        "voice\\other",
    ],
)
def test_voice_identifiers_cannot_redirect_sdk_operations(voice_id):
    from pydantic import ValidationError

    from elevenlabs_mcp.contracts import CastVoice

    with pytest.raises(ValidationError):
        CastVoice(voice_id=voice_id)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, content=b"audio")

    provider = ElevenLabsProvider(
        "fixture", httpx.Client(transport=httpx.MockTransport(handle))
    )
    p = plan()
    unsafe = p.requests[0].model_copy(
        update={
            "chunk": p.requests[0].chunk.model_copy(update={"voice_ids": (voice_id,)})
        }
    )
    with pytest.raises(ProviderError, match="UNSUPPORTED_SETTINGS"):
        list(provider.generate(unsafe, p.resolved_options))
    assert calls == []
