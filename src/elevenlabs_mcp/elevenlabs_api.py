import logging
import os
import sys
import time
import requests
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, TypedDict
from uuid import UUID, uuid4

log_level = os.getenv("ELEVENLABS_LOG_LEVEL", "ERROR").upper()
valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if log_level not in valid_levels:
    log_level = "ERROR"
    print("Invalid log level; using ERROR.", file=sys.stderr)

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VoiceData(TypedDict):
    voice_id: str
    name: str
    category: str
    labels: Dict[str, str]
    description: str
    preview_url: str
    high_quality_base_model_ids: List[str]


class _MissingAPIKeyError(ValueError):
    """Credential absence detected before provider dispatch."""


class _NonRetryableProviderError(RuntimeError):
    """A deterministic provider rejection that must not be replayed."""


class PartialGenerationError(RuntimeError):
    """A required script part failed after zero or more parts were retained."""

    def __init__(
        self,
        partial_output_file: Optional[str],
        completed_parts: int,
        failed_part_indexes: tuple[int, ...],
    ) -> None:
        self.partial_output_file = partial_output_file
        self.completed_parts = completed_parts
        self.failed_part_indexes = failed_part_indexes
        indexes = ", ".join(str(index) for index in failed_part_indexes)
        partial = partial_output_file or "none"
        super().__init__(
            f"Audio generation failed after {completed_parts} completed parts; "
            f"failed part indexes: {indexes}; partial audio: {partial}"
        )


from pydub import AudioSegment
import io
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_legacy_retry_wait = wait_exponential(multiplier=1, min=4, max=10)


class _RetryableRateLimitError(RuntimeError):
    def __init__(self, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__("Provider rate limit")


class UpstreamOutcomeUnknownError(RuntimeError):
    """Synthesis may have reached the provider and must not be replayed blindly."""

    def __init__(self, operation: str, cause_type: str) -> None:
        self.operation = operation
        self.cause_type = cause_type
        self.retryable = False
        super().__init__(
            f"{operation} upstream outcome is unknown after {cause_type}; "
            "retryable: false"
        )


def _retry_wait(retry_state):
    exception = retry_state.outcome.exception()
    if isinstance(exception, _RetryableRateLimitError):
        retry_after = exception.retry_after_seconds
        if retry_after is not None:
            return retry_after
    return _legacy_retry_wait(retry_state)


def _retry_after_seconds(response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if 0.0 <= seconds <= 10.0 else None

class ElevenLabsAPI:
    # Add model list as class constant
    MODELS = {
        "eleven_multilingual_v2": {"description": "Our most lifelike model with rich emotional expression", "languages": "32",
                                   "supports_stitching": True, "supports_style": True, "wait_time": 0.1},
        "eleven_flash_v2_5": {"description": "Ultra-fast model optimized for real-time use (~75ms†)", "languages": "32",
                              "supports_stitching": False, "supports_style": False, "wait_time": 0.1},
        "eleven_flash_v2": {"description": "Ultra-fast model optimized for real-time use (~75ms†)", "languages": "English",
                             "supports_stitching": False, "supports_style": False, "wait_time": 0.1}
    }

    @retry(
        stop=stop_after_attempt(3),
        wait=_retry_wait,
        retry=retry_if_not_exception_type(
            (
                _MissingAPIKeyError,
                _NonRetryableProviderError,
                UpstreamOutcomeUnknownError,
            )
        ),
    )
    def get_voices(self) -> List[VoiceData]:
        """Fetch available voices from ElevenLabs API"""
        api_key = self._require_api_key()
        headers = {
            "Accept": "application/json",
            "xi-api-key": api_key
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/voices",
                headers=headers,
                timeout=(5.0, 60.0),
            )
        except requests.exceptions.RequestException as error:
            error_message = f"Voice metadata network error: {type(error).__name__}"
            logging.error(error_message)
            raise Exception(error_message)
        
        if response.status_code == 200:
            try:
                voices_data = response.json()["voices"]
                return [
                    {
                        "voice_id": voice["voice_id"],
                        "name": voice["name"],
                        "category": voice.get("category", ""),
                        "labels": voice.get("labels", {}),
                        "description": voice.get("description", ""),
                        "preview_url": voice.get("preview_url", ""),
                        "high_quality_base_model_ids": voice.get("high_quality_base_model_ids", [])
                    }
                    for voice in voices_data
                ]
            except Exception as error:
                error_message = (
                    f"Voice metadata response invalid: {type(error).__name__}"
                )
                logging.error(error_message)
                raise Exception(error_message)
        else:
            error_message = f"Voice metadata request failed with status {response.status_code}"
            if response.status_code == 429:
                raise _RetryableRateLimitError(_retry_after_seconds(response))
            if response.status_code in {400, 401, 403, 404, 422}:
                raise _NonRetryableProviderError(error_message)
            raise Exception(error_message)

    def __init__(self, environ: Mapping[str, str] | None = None):
        environment = os.environ if environ is None else environ
        self.api_key = environment.get("ELEVENLABS_API_KEY") or None

        self.voice_id = environment.get("ELEVENLABS_VOICE_ID") or "iEw1wkYocsNy7I7pteSN"
        self.model_id = environment.get("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
        
        logging.info(f"Initializing ElevenLabsAPI with model_id: {self.model_id}")
        
        # Add validation for model_id
        if self.model_id not in self.MODELS:
            logging.error(f"Invalid model_id: {self.model_id}. Valid models: {list(self.MODELS.keys())}")
            raise ValueError(f"Invalid model_id: {self.model_id}. Must be one of {list(self.MODELS.keys())}")
        self.stability = float(environment.get("ELEVENLABS_STABILITY", "0.5"))
        self.similarity_boost = float(
            environment.get("ELEVENLABS_SIMILARITY_BOOST", "0.75")
        )
        self.style = float(environment.get("ELEVENLABS_STYLE", "0.1"))
        self.base_url = "https://api.elevenlabs.io/v1"

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise _MissingAPIKeyError("ELEVENLABS_API_KEY environment variable not set")
        return self.api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=_retry_wait,
        retry=retry_if_not_exception_type(
            (
                _MissingAPIKeyError,
                _NonRetryableProviderError,
                UpstreamOutcomeUnknownError,
            )
        ),
    )
    def generate_audio_segment(self, text: str, voice_id: str, output_file: Optional[str] = None,
                      previous_text: Optional[str] = None, next_text: Optional[str] = None,
                      previous_request_ids: Optional[List[str]] = None, debug_info: Optional[List[str]] = None) -> tuple[bytes, str]:
        """Generate audio using specified voice with context conditioning"""
        api_key = self._require_api_key()
        headers = {
            "Accept": "application/json",
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost
            }
        }

        if self.MODELS[self.model_id]["supports_style"]:
            data["voice_settings"]["style"] = self.style

        # Add context conditioning if model supports it
        if self.MODELS[self.model_id]["supports_stitching"]:
            if previous_text is not None:
                data["previous_text"] = previous_text
            if next_text is not None:
                data["next_text"] = next_text
            if previous_request_ids:
                data["previous_request_ids"] = previous_request_ids[-3:]  # Maximum of 3 previous IDs
        
        logging.info("Generating audio for text length: %s chars", len(text))
        logging.debug(f"Generation parameters: stability={self.stability}, similarity_boost={self.similarity_boost}, model={self.model_id}")
        
        try:
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                json=data,
                headers=headers,
                timeout=(5.0, 60.0),
            )
            
            logging.debug(f"API response status: {response.status_code}")
            
            if response.status_code == 200:
                logging.info("Audio generation successful")
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                return response.content, response.headers["request-id"]
            else:
                error_message = (
                    f"Audio provider request failed with status {response.status_code}"
                )
                logging.error(f"API error response: {response.status_code}")
                if response.status_code == 429:
                    raise _RetryableRateLimitError(_retry_after_seconds(response))
                if response.status_code in {400, 401, 403, 404, 422}:
                    raise _NonRetryableProviderError(error_message)
                raise Exception(error_message)
        except requests.exceptions.ConnectTimeout as e:
            error_message = f"Network error during API call: {type(e).__name__}"
            logging.error(error_message)
            raise Exception(error_message)
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            raise UpstreamOutcomeUnknownError("synthesis", type(e).__name__)
        except requests.exceptions.RequestException as e:
            raise UpstreamOutcomeUnknownError("synthesis", type(e).__name__)

    def generate_full_audio(
        self,
        script_parts: List[Dict],
        output_dir: Path,
        output_id: Optional[str] = None,
    ) -> tuple[str, List[str], int]:
        """Generate audio for multiple parts using request stitching. Returns tuple of (output_file_path, debug_info, completed_parts)"""
        self._require_api_key()
        canonical_output_id = str(uuid4()) if output_id is None else str(UUID(output_id))
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        debug_info = []
        debug_info.append("ElevenLabsAPI - Starting generate_full_audio")
        debug_info.append(f"Script part count: {len(script_parts)}")
        
        # Initialize segments list and request IDs tracking
        segments = []
        previous_request_ids = []
        failed_part_indexes: list[int] = []
        completed_parts = 0
        
        all_texts = []
        for part in script_parts:
            text = str(part.get('text', ''))
            all_texts.append(text)
        
        for i, part in enumerate(script_parts):
            debug_info.append(f"Processing part index: {i}")
            part_voice_id = part.get('voice_id')
            if not part_voice_id:
                part_voice_id = self.voice_id
            text = str(part.get('text', ''))
            if not text:
                continue
                
            
            # Determine previous and next text for context
            is_first = i == 0
            is_last = i == len(script_parts) - 1
            
            previous_text = None if is_first else " ".join(all_texts[:i])
            next_text = None if is_last else " ".join(all_texts[i + 1:])
            
            try:
                logging.info(f"Processing part {i+1}/{len(script_parts)}")
                logging.info(f"Text length: {len(text)} chars")
                logging.debug(f"Context - Previous text: {'Yes' if previous_text else 'No'}, Next text: {'Yes' if next_text else 'No'}")
                
                # Generate audio with context conditioning
                audio_content, request_id = self.generate_audio_segment(
                    text=text,
                    voice_id=part_voice_id,
                    previous_text=previous_text,
                    next_text=next_text,
                    previous_request_ids=previous_request_ids,
                    debug_info=debug_info
                )
                
                # Add request ID to history
                previous_request_ids.append(request_id)
                
                # Convert audio content to AudioSegment and add to segments
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
                segments.append(audio_segment)
                completed_parts += 1
                debug_info.append(f"Generated audio for part {i}")

                # Wait for the specified wait_time
                time.sleep(self.MODELS[self.model_id]["wait_time"])
            except Exception as e:
                debug_info.append(f"Part {i} failed: {type(e).__name__}")
                failed_part_indexes.append(i)
                continue
        
        # Combine all segments
        if segments:
            output_prefix = "partial_audio" if failed_part_indexes else "full_audio"
            output_file = output_dir / f"{output_prefix}_{canonical_output_id}.mp3"
            final_audio = segments[0]
            for segment in segments[1:]:
                final_audio = final_audio + segment
            
            # Export to an owned temporary file, then atomically publish without
            # replacing an existing artifact for the same job identity.
            with NamedTemporaryFile(
                dir=output_dir,
                prefix=f".full_audio_{canonical_output_id}_",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                final_audio.export(temporary_path, format="mp3")
                os.link(temporary_path, output_file)
            finally:
                temporary_path.unlink(missing_ok=True)

            if not failed_part_indexes:
                logging.debug("All parts generated successfully")
                debug_info.append("All parts generated successfully")
            
            debug_info.append(f"Model: {self.model_id}")
            logging.debug(f"Model: {self.model_id}")
            
            if failed_part_indexes:
                raise PartialGenerationError(
                    partial_output_file=str(output_file),
                    completed_parts=completed_parts,
                    failed_part_indexes=tuple(failed_part_indexes),
                )
            return str(output_file), debug_info, completed_parts
        if failed_part_indexes:
            raise PartialGenerationError(
                partial_output_file=None,
                completed_parts=0,
                failed_part_indexes=tuple(failed_part_indexes),
            )
        else:
            error_msg = "\n".join([
                "No audio segments were generated. Debug info:",
                *debug_info
            ])
            logging.error("No audio segments were generated. Debug info: %s", debug_info)
            raise Exception(error_msg)
