import asyncio
import base64
import json
import logging
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote

import mcp.server.stdio
from mcp import types
from mcp.server import NotificationOptions
from mcp.server.models import InitializationOptions

from .config import (
    Settings,
    environment_from_dotenv,
    select_database_path,
    settings_from_environment,
)
from .database import Database
from .elevenlabs_api import (
    ElevenLabsAPI,
    PartialGenerationError,
    UpstreamOutcomeUnknownError,
)
from .models import AudioJob
from .protocol import ProtocolServer
from .version import package_version

logger = logging.getLogger(__name__)

log_level = os.getenv("ELEVENLABS_LOG_LEVEL", "ERROR").upper()
valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if log_level not in valid_levels:
    log_level = "ERROR"
    print("Invalid log level; using ERROR.", file=sys.stderr)

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ElevenLabsServer:
    def __init__(
        self,
        settings: Settings | None = None,
        environ: Mapping[str, str] | None = None,
        *,
        enable_revival: bool = True,
        revival_service=None,
    ):
        launch_cwd = settings.launch_cwd if settings is not None else Path.cwd()
        runtime_environment = environment_from_dotenv(
            os.environ if environ is None else environ,
            launch_cwd,
        )
        self.settings = settings or settings_from_environment(
            runtime_environment, launch_cwd
        )
        self.server = ProtocolServer("elevenlabs-server")
        self.api = ElevenLabsAPI(runtime_environment)
        self.output_dir = self.settings.output_dir
        self.db = Database(self.settings.database_path)
        self._generation_lock = asyncio.Lock()
        self.revival_service = revival_service
        if enable_revival and revival_service is None:
            from .jobs import VoiceoverService
            from .provider import ElevenLabsProvider

            self.revival_service = VoiceoverService(
                self.settings.database_path,
                self.output_dir,
                ElevenLabsProvider(runtime_environment.get("ELEVENLABS_API_KEY")),
            )
        if self.revival_service is not None:
            from .mcp_app import RevivalTools

            self.server.revival = RevivalTools(self.revival_service, self)

        # Set up handlers
        self.setup_tools()
        self.setup_resources()
        # self.setup_notifications()

    async def _generate_full_audio(
        self, script_parts: list[dict], job_id: str
    ) -> tuple[str, list[str], int]:
        async with self._generation_lock:
            return await asyncio.to_thread(
                self.api.generate_full_audio,
                script_parts,
                self.output_dir,
                output_id=job_id,
            )

    async def initialize(self) -> None:
        """Initialize server components."""
        select_database_path(
            self.settings,
            Path(__file__).resolve().parents[2],
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.revival_service is not None:
            await self.revival_service.start()
            try:
                await self.db.initialize()
            except BaseException:
                await self.revival_service.close()
                raise
            return
        await self.db.initialize()

        # Initialize voices cache
        try:
            _voices, needs_refresh = await self.db.get_voices()
            if needs_refresh and self.api.api_key:
                logger.info("Fetching initial voices data")
                fresh_voices = await asyncio.to_thread(self.api.get_voices)
                await self.db.upsert_voices(fresh_voices)
                logger.info(f"Cached {len(fresh_voices)} voices")
            elif needs_refresh:
                logger.info(
                    "Skipping initial voices refresh: API key is not configured"
                )
        except Exception as e:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
            logger.error("Error initializing voices cache: %s", type(e).__name__)

    def parse_script(
        self, script_json: str
    ) -> tuple[list[dict[str, str | None]], list[str]]:
        """
        Parse the input into a list of script parts and collect debug information.
        Accepts:
        1. A JSON string with a script array containing dialogue parts
        2. Plain text to be converted to speech

        Each dialogue part should have:
        - text (required): The text to speak
        - voice_id (optional): The voice to use
        - actor (optional): The actor/character name

        Args:
            script_json: Input text or JSON string

        Returns:
            tuple containing:
                - list of parsed script parts
                - list of debug information strings
        """
        debug_info = []

        script_array = []

        # Remove any leading/trailing whitespace
        script_json = script_json.strip()

        try:
            # Try to parse as JSON first
            if script_json.startswith("["):
                # Direct array of script parts
                script_array = json.loads(script_json)
            elif script_json.startswith("{"):
                # Object with script array
                script_data = json.loads(script_json)
                script_array = script_data.get("script", [])
            else:
                # Treat as plain text if not JSON formatted
                script_array = [{"text": script_json}]
        except json.JSONDecodeError:
            # If JSON parsing fails and input looks like JSON, raise error
            if script_json.startswith(("{", "[")):
                debug_info.append("JSON parsing failed")
                raise RuntimeError("Invalid JSON format")
            # Otherwise treat as plain text
            debug_info.append("Parsed plain text input")
            script_array = [{"text": script_json}]

        script_parts = []
        for index, part in enumerate(script_array):
            if not isinstance(part, dict):
                debug_info.append(f"Skipped non-object script part at index {index}")
                continue

            text = part.get("text", "").strip()
            if not text:
                debug_info.append("Missing or empty text field")
                raise RuntimeError("Missing required field 'text'")

            new_part = {
                "text": text,
                "voice_id": part.get("voice_id"),
                "actor": part.get("actor"),
            }
            debug_info.append(f"Parsed script part at index {index}")
            script_parts.append(new_part)

        debug_info.append(f"Parsed script part count: {len(script_parts)}")
        return script_parts, debug_info

    def setup_resources(self) -> None:
        """Set up MCP resources."""

        @self.server.list_resource_templates()
        async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
            return [
                types.ResourceTemplate(
                    uri_template="voiceover://history/{job_id}",
                    name="Voiceover Job History",
                    description="Access voiceover job history. Provide job_id for specific job or omit for all jobs.",
                    mime_type="application/json",
                ),
                types.ResourceTemplate(
                    uri_template="voiceover://voices",
                    name="Available Voices",
                    description="Access list of available ElevenLabs voices with metadata",
                    mime_type="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            uri_str = str(uri)

            if uri_str == "voiceover://voices":
                try:
                    # Get voices from cache
                    voices, needs_refresh = await self.db.get_voices()

                    # Refresh cache if needed
                    if needs_refresh:
                        try:
                            fresh_voices = await asyncio.to_thread(self.api.get_voices)
                            await self.db.upsert_voices(fresh_voices)
                            voices = fresh_voices
                        except Exception as e:
                            logger.error(
                                "Error refreshing voices: %s", type(e).__name__
                            )
                            # Continue with cached data if refresh fails
                            if not voices:
                                raise  # Re-raise if we have no data at all

                    # Ensure default voice is marked
                    for voice in voices:
                        voice["is_default"] = voice["voice_id"] == self.api.voice_id

                    return json.dumps(voices, indent=2)
                except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                    return json.dumps({"error": "Voice metadata unavailable"}, indent=2)

            if not uri_str.startswith("voiceover://history"):
                raise ValueError(f"Invalid resource URI: {uri_str}")

            try:
                # Extract job_id if present
                parts = uri_str.split("/")
                logger.info(f"Parts: {parts}")
                if len(parts) > 3 and unquote(parts[3]) != "{job_id}":
                    job_id = parts[3]
                    job = await self.db.get_job(job_id)
                    if not job:
                        return json.dumps({"error": "Job not found"}, indent=2)
                    jobs = [job]
                else:
                    jobs = await self.db.get_all_jobs()

                # Convert jobs to JSON
                jobs_data = [job.to_dict() for job in jobs]
                return json.dumps(jobs_data, indent=2)

            except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                return json.dumps({"error": "History unavailable"}, indent=2)

    def setup_tools(self) -> None:
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="generate_audio_simple",
                    description="Generate audio from plain text using default voice settings",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Plain text to convert to audio",
                            },
                            "voice_id": {
                                "type": "string",
                                "description": "Optional voice ID to use for generation",
                            },
                        },
                        "required": ["text"],
                    },
                ),
                types.Tool(
                    name="generate_audio_script",
                    description="""Generate audio from a structured script with multiple voices and actors. 
                    Accepts either:
                    1. Plain text string
                    2. JSON string with format: {
                        "script": [
                            {
                                "text": "Text to speak",
                                "voice_id": "optional-voice-id",
                                "actor": "optional-actor-name"
                            },
                            ...
                        ]
                    }""",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "JSON string containing script array or plain text. For JSON format, provide an object with a 'script' array containing objects with 'text' (required), 'voice_id' (optional), and 'actor' (optional) fields.",
                            }
                        },
                        "required": ["script"],
                    },
                ),
                types.Tool(
                    name="delete_job",
                    description="Delete a voiceover job and its associated files",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "job_id": {
                                "type": "string",
                                "description": "ID of the job to delete",
                            }
                        },
                        "required": ["job_id"],
                    },
                ),
                types.Tool(
                    name="get_audio_file",
                    description="Get the audio file content for a specific job",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "job_id": {
                                "type": "string",
                                "description": "ID of the job to get audio file for",
                            }
                        },
                        "required": ["job_id"],
                    },
                ),
                types.Tool(
                    name="list_voices",
                    description="Get a list of all available ElevenLabs voices with metadata",
                    input_schema={
                        "type": "object",
                        "properties": {},  # No parameters needed
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_voiceover_history",
                    description="Get voiceover job history. Optionally specify a job ID for a specific job.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "job_id": {
                                "type": "string",
                                "description": "Optional job ID to get details for a specific job",
                            }
                        },
                        "required": [],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict
        ) -> list[types.TextContent | types.EmbeddedResource]:
            debug_info: list[str] = []
            try:
                if name == "generate_audio_simple":
                    debug_info.append("Processing simple audio request")

                    text = arguments.get("text", "").strip()
                    voice_id = arguments.get("voice_id")

                    if not text:
                        raise ValueError("Text cannot be empty")

                    script_parts = [{"text": text, "voice_id": voice_id}]

                    # Create job record
                    job_id = str(uuid.uuid4())
                    job = AudioJob.create(
                        id=job_id,
                        status="pending",
                        script_parts=script_parts,
                        total_parts=1,
                    )
                    await self.db.insert_job(job)
                    debug_info.append(f"Created job record: {job_id}")

                    try:
                        job.status = "processing"
                        await self.db.update_job(job)

                        # # Send progress notification
                        # if hasattr(self.server, 'session'):
                        #     await self.server.session.send_notification({
                        #         "method": "notifications/progress",
                        #         "params": {
                        #             "progressToken": str(job.id),
                        #             "progress": {
                        #                 "kind": "begin",
                        #                 "message": "Starting audio generation"
                        #             }
                        #         }
                        #     })

                        (
                            output_file,
                            api_debug_info,
                            completed_parts,
                        ) = await self._generate_full_audio(script_parts, job_id)
                        del api_debug_info
                        debug_info.append(
                            f"Generated part count: {completed_parts}/{len(script_parts)}"
                        )

                        job.status = "completed"
                        job.output_file = str(output_file)
                        job.completed_parts = completed_parts
                        await self.db.update_job(job)

                        # # Send completion notification
                        # if hasattr(self.server, 'session'):
                        #     await self.server.session.send_notification({
                        #         "method": "notifications/progress",
                        #         "params": {
                        #             "progressToken": str(job.id),
                        #             "progress": {
                        #                 "kind": "end",
                        #                 "message": "Audio generation completed"
                        #             }
                        #         }
                        #     })
                    except UpstreamOutcomeUnknownError as e:
                        job.output_file = e.partial_output_file
                        job.completed_parts = e.completed_parts
                        job.status = "failed"
                        job.error = str(e)
                        await self.db.update_job(job)
                        raise
                    except PartialGenerationError as e:
                        job.status = "failed"
                        job.output_file = e.partial_output_file
                        job.completed_parts = e.completed_parts
                        job.error = str(e)
                        await self.db.update_job(job)
                        raise
                    except Exception:
                        job.status = "failed"
                        job.error = "Audio generation failed"
                        await self.db.update_job(job)
                        raise

                    # Read the generated audio file and encode it as base64
                    audio_bytes = await asyncio.to_thread(Path(output_file).read_bytes)
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                    # Generate unique URI for the resource
                    filename = Path(output_file).name
                    resource_uri = f"audio://{filename}"

                    # Return both a status message and the audio file content
                    return [
                        types.TextContent(
                            type="text",
                            text="\n".join(
                                [
                                    "Audio generation successful. Debug info:",
                                    *debug_info,
                                ]
                            ),
                        ),
                        types.EmbeddedResource(
                            type="resource",
                            resource=types.BlobResourceContents(
                                uri=resource_uri,
                                blob=audio_base64,
                                mime_type="audio/mpeg",
                            ),
                        ),
                    ]

                elif name == "generate_audio_script":
                    script_json = arguments.get("script", "{}")
                    script_parts, parse_debug_info = self.parse_script(script_json)
                    debug_info.extend(parse_debug_info)

                    # Create job record
                    job_id = str(uuid.uuid4())
                    job = AudioJob.create(
                        id=job_id,
                        status="pending",
                        script_parts=script_parts,
                        total_parts=len(script_parts),
                    )
                    await self.db.insert_job(job)
                    debug_info.append(f"Created job record: {job_id}")

                    try:
                        job.status = "processing"
                        await self.db.update_job(job)

                        (
                            output_file,
                            api_debug_info,
                            completed_parts,
                        ) = await self._generate_full_audio(script_parts, job_id)
                        del api_debug_info
                        debug_info.append(
                            f"Generated part count: {completed_parts}/{len(script_parts)}"
                        )

                        job.status = "completed"
                        job.output_file = str(output_file)
                        job.completed_parts = completed_parts
                        await self.db.update_job(job)
                    except UpstreamOutcomeUnknownError as e:
                        job.output_file = e.partial_output_file
                        job.completed_parts = e.completed_parts
                        job.status = "failed"
                        job.error = str(e)
                        await self.db.update_job(job)
                        raise
                    except PartialGenerationError as e:
                        job.status = "failed"
                        job.output_file = e.partial_output_file
                        job.completed_parts = e.completed_parts
                        job.error = str(e)
                        await self.db.update_job(job)
                        raise
                    except Exception:
                        job.status = "failed"
                        job.error = "Audio generation failed"
                        await self.db.update_job(job)
                        raise

                    # Read the generated audio file and encode it as base64
                    audio_bytes = await asyncio.to_thread(Path(output_file).read_bytes)
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                    # Generate unique URI for the resource
                    filename = Path(output_file).name
                    resource_uri = f"audio://{filename}"

                    # Return both a status message and the audio file content
                    return [
                        types.TextContent(
                            type="text",
                            text="\n".join(
                                [
                                    "Audio generation successful. Debug info:",
                                    *debug_info,
                                ]
                            ),
                        ),
                        types.EmbeddedResource(
                            type="resource",
                            resource=types.BlobResourceContents(
                                uri=resource_uri,
                                blob=audio_base64,
                                mime_type="audio/mpeg",
                            ),
                        ),
                    ]

                elif name == "delete_job":
                    job_id = arguments.get("job_id")
                    if not job_id:
                        raise ValueError("job_id is required")

                    # Get job to check if it exists and get file path
                    job = await self.db.get_job(job_id)
                    if not job:
                        return [
                            types.TextContent(
                                type="text", text=f"Job {job_id} not found"
                            )
                        ]

                    # Delete associated audio file if it exists
                    if job.output_file:
                        try:
                            output_path = Path(job.output_file)
                            if output_path.exists():
                                output_path.unlink()
                        except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                            return [
                                types.TextContent(
                                    type="text", text="Error deleting audio file"
                                )
                            ]

                    # Delete job from database
                    await self.db.delete_job(job_id)
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Successfully deleted job {job_id} and associated files",
                        )
                    ]

                elif name == "list_voices":
                    try:
                        # Get voices from cache
                        voices, needs_refresh = await self.db.get_voices()

                        # Refresh cache if needed
                        if needs_refresh:
                            try:
                                fresh_voices = await asyncio.to_thread(
                                    self.api.get_voices
                                )
                                await self.db.upsert_voices(fresh_voices)
                                voices = fresh_voices
                            except Exception as e:
                                logger.error(
                                    "Error refreshing voices: %s", type(e).__name__
                                )
                                # Continue with cached data if refresh fails
                                if not voices:
                                    raise  # Re-raise if we have no data at all

                        # Ensure default voice is marked
                        for voice in voices:
                            voice["is_default"] = voice["voice_id"] == self.api.voice_id

                        return [
                            types.TextContent(
                                type="text", text=json.dumps(voices, indent=2)
                            )
                        ]
                    except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                        return [
                            types.TextContent(
                                type="text",
                                text=json.dumps(
                                    {"error": "Voice metadata unavailable"}, indent=2
                                ),
                            )
                        ]

                elif name == "get_voiceover_history":
                    try:
                        job_id = arguments.get("job_id")
                        if job_id:
                            job = await self.db.get_job(job_id)
                            if not job:
                                return [
                                    types.TextContent(
                                        type="text",
                                        text=json.dumps(
                                            {"error": "Job not found"}, indent=2
                                        ),
                                    )
                                ]
                            jobs = [job]
                        else:
                            jobs = await self.db.get_all_jobs()

                        # Convert jobs to JSON
                        jobs_data = [job.to_dict() for job in jobs]
                        return [
                            types.TextContent(
                                type="text", text=json.dumps(jobs_data, indent=2)
                            )
                        ]

                    except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                        return [
                            types.TextContent(
                                type="text",
                                text=json.dumps(
                                    {"error": "History unavailable"}, indent=2
                                ),
                            )
                        ]

                elif name == "get_audio_file":
                    job_id = arguments.get("job_id")
                    if not job_id:
                        raise ValueError("job_id is required")

                    # Get job to check if it exists and get file path
                    job = await self.db.get_job(job_id)
                    if not job:
                        return [
                            types.TextContent(
                                type="text", text=f"Job {job_id} not found"
                            )
                        ]

                    if not job.output_file:
                        return [
                            types.TextContent(
                                type="text",
                                text=f"No output file found for job {job_id}",
                            )
                        ]

                    # Check if file exists
                    output_path = Path(job.output_file)
                    if not output_path.exists():
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Output file not found at {job.output_file}",
                            )
                        ]

                    # Read the audio file and encode it as base64
                    audio_bytes = await asyncio.to_thread(Path(output_path).read_bytes)
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                    # Return the audio file content
                    return [
                        types.EmbeddedResource(
                            type="resource",
                            resource=types.BlobResourceContents(
                                uri=f"audio://{output_path.name}",
                                blob=audio_base64,
                                mime_type="audio/mpeg",
                            ),
                        )
                    ]

                else:
                    return [
                        types.TextContent(type="text", text=f"Unknown tool: {name}")
                    ]

            except Exception as e:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                error_detail = (
                    str(e)
                    if isinstance(
                        e, (PartialGenerationError, UpstreamOutcomeUnknownError)
                    )
                    else "Audio generation failed"
                )
                error_msg = "\n".join(
                    [
                        "Error generating audio. Debug info:",
                        *debug_info,
                        f"Error: {error_detail}",
                    ]
                )
                return [types.TextContent(type="text", text=error_msg)]

    def setup_notifications(self) -> None:
        @self.server.progress_notification()
        async def handle_progress(request_id, progress, total):
            logger.info("Received progress notification for %s", request_id)

    async def run(self) -> None:
        """Run the server"""
        try:
            await self.initialize()
        except Exception as e:
            logger.error("Server initialization failed: %s", type(e).__name__)
            raise
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="elevenlabs-server",
                        server_version=package_version(),
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        finally:
            if self.revival_service is not None:
                await self.revival_service.close()


def main() -> None:
    """Entry point for the server"""
    server = ElevenLabsServer(enable_revival=True)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
