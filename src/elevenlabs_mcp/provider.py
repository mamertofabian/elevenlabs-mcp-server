"""Official ElevenLabs SDK boundary; synthesis has one wire attempt per reservation."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterator
from typing import Literal, NotRequired, Protocol, TypedDict

import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs.core import ApiError, RequestOptions
from elevenlabs.types import DialogueInput, VoiceSettings

from .contracts import PlannedRequest, VoiceoverOptions

TTS_MODELS = {
    "eleven_multilingual_v2",
    "eleven_flash_v2_5",
    "eleven_flash_v2",
    "eleven_turbo_v2_5",
}


class GenerationOptions(TypedDict):
    model_id: str
    output_format: Literal["mp3_44100_128"]
    request_options: RequestOptions
    seed: NotRequired[int]
    language_code: NotRequired[str]


class ProviderError(RuntimeError):
    def __init__(self, code: str, *, uncertain: bool = False):
        self.code = code
        self.uncertain = uncertain
        super().__init__(code)


class SpeechProvider(Protocol):
    def check_ready(self, options: VoiceoverOptions) -> None: ...
    def generate(
        self, request: PlannedRequest, options: VoiceoverOptions
    ) -> Iterator[bytes]: ...
    def close(self) -> None: ...


def validate_profile(options: VoiceoverOptions) -> None:
    if options.engine == "tts" and options.model_id not in TTS_MODELS:
        raise ProviderError("UNSUPPORTED_MODEL")
    if (
        options.engine == "tts"
        and options.model_id == "eleven_flash_v2"
        and options.language_code not in {None, "en"}
    ):
        raise ProviderError("UNSUPPORTED_SETTINGS")
    if (
        options.voice_settings
        and options.model_id.startswith("eleven_flash")
        and options.voice_settings.style is not None
    ):
        raise ProviderError("UNSUPPORTED_SETTINGS")


class ElevenLabsProvider:
    def __init__(self, api_key: str | None, http_client: httpx.Client | None = None):
        self.api_key = api_key
        self.context_id = hashlib.sha256((api_key or "").encode()).hexdigest()
        self._http = http_client
        self._owned = http_client is None
        self._client: ElevenLabs | None = None
        self._client_lock = threading.Lock()
        self.last_request_id: str | None = None

    def check_ready(self, options: VoiceoverOptions) -> None:
        validate_profile(options)
        if not self.api_key:
            raise ProviderError("API_KEY_MISSING")

    def _sdk(self) -> ElevenLabs:
        if not self.api_key:
            raise ProviderError("API_KEY_MISSING")
        with self._client_lock:
            if self._client is None:
                self._http = self._http or httpx.Client(
                    timeout=httpx.Timeout(60, connect=5),
                    follow_redirects=False,
                    trust_env=False,
                )
                self._client = ElevenLabs(
                    api_key=self.api_key,
                    httpx_client=self._http,
                    timeout=60,
                    follow_redirects=False,
                )
        return self._client

    def generate(
        self, request: PlannedRequest, options: VoiceoverOptions
    ) -> Iterator[bytes]:
        self.check_ready(options)
        voice_ids = (
            *request.chunk.voice_ids,
            *(item.voice_id for item in request.chunk.fragments),
        )
        if any(
            re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity) is None
            for identity in voice_ids
        ):
            raise ProviderError("UNSUPPORTED_SETTINGS")
        self.last_request_id = None
        sdk = self._sdk()
        common: GenerationOptions = {
            "model_id": options.model_id,
            "output_format": "mp3_44100_128",
            "request_options": {
                "max_retries": 0,
                "timeout_in_seconds": 60,
                "chunk_size": 65536,
            },
        }
        if request.seed is not None:
            common["seed"] = request.seed
        if options.language_code:
            common["language_code"] = options.language_code
        try:
            if options.engine == "dialogue":
                stream = sdk.text_to_dialogue.with_raw_response.convert(
                    inputs=[
                        DialogueInput(text=f.text, voice_id=f.voice_id)
                        for f in request.chunk.fragments
                    ],
                    **common,
                )
            else:
                settings = options.voice_settings
                stream = sdk.text_to_speech.with_raw_response.convert(
                    voice_id=request.chunk.voice_ids[0],
                    text="".join(f.text for f in request.chunk.fragments),
                    voice_settings=VoiceSettings(
                        **settings.model_dump(exclude_none=True)
                    )
                    if settings
                    else None,
                    **common,
                )
            with stream as response:
                request_id = response.headers.get("request-id")
                self.last_request_id = (
                    request_id if request_id and len(request_id) <= 128 else None
                )
                yield from response.data
        except ApiError as error:
            status = error.status_code or 0
            code = {
                401: "AUTHENTICATION_FAILED",
                403: "AUTHENTICATION_FAILED",
                429: "RATE_LIMITED",
                400: "PROVIDER_VALIDATION_FAILED",
                422: "PROVIDER_VALIDATION_FAILED",
            }.get(status, "UPSTREAM_OUTCOME_UNKNOWN")
            raise ProviderError(
                code, uncertain=code == "UPSTREAM_OUTCOME_UNKNOWN"
            ) from None
        except httpx.ConnectTimeout:
            raise ProviderError("CONNECTION_FAILED") from None
        except httpx.HTTPError:
            raise ProviderError("UPSTREAM_OUTCOME_UNKNOWN", uncertain=True) from None

    def search_voices(self, query: str, limit: int, cursor: str | None):
        try:
            page = self._sdk().voices.search(
                search=query,
                page_size=limit,
                next_page_token=cursor,
                request_options={"max_retries": 0},
            )
            return {
                "voices": [
                    {
                        "voice_id": v.voice_id,
                        "name": v.name,
                        "category": v.category,
                        "preview_url": v.preview_url,
                        "labels": v.labels or {},
                        "description": v.description or "",
                        "high_quality_base_model_ids": v.high_quality_base_model_ids
                        or [],
                    }
                    for v in page.voices
                ],
                "next_cursor": page.next_page_token,
                "stale": False,
            }
        except (ApiError, httpx.HTTPError):
            raise ProviderError("METADATA_UNAVAILABLE") from None

    def list_models(self):
        try:
            return {
                "models": [
                    {
                        "model_id": m.model_id,
                        "name": m.name,
                        "supported_engines": ["tts"]
                        if m.model_id in TTS_MODELS
                        else ["dialogue"]
                        if m.model_id == "eleven_v3"
                        else [],
                    }
                    for m in self._sdk().models.list(request_options={"max_retries": 0})
                ]
            }
        except (ApiError, httpx.HTTPError):
            raise ProviderError("METADATA_UNAVAILABLE") from None

    def close(self):
        if self._owned and self._http:
            self._http.close()
            self._client = None
            self._http = None
