"""ElevenLabs MCP Server package."""

from .version import package_version

__version__ = package_version()

from .elevenlabs_api import ElevenLabsAPI
from .models import AudioJob, ScriptPart
from .server import ElevenLabsServer, main

__all__ = ["AudioJob", "ElevenLabsAPI", "ElevenLabsServer", "ScriptPart", "main"]
