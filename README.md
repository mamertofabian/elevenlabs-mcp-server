# ElevenLabs MCP Server

A Model Context Protocol (MCP) server that integrates with ElevenLabs text-to-speech API.

<a href="https://glama.ai/mcp/servers/leukzvus7o"><img width="380" height="200" src="https://glama.ai/mcp/servers/leukzvus7o/badge" alt="ElevenLabs Server MCP server" /></a>

## Features

- Generate audio from text using ElevenLabs API
- Support for multiple voices and script parts

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```
3. Copy `.env.example` to `.env` and fill in your ElevenLabs credentials

## MCP Settings Configuration

Add the following configuration to your MCP settings file (e.g., `cline_mcp_settings.json` for Claude Desktop):

```json
{
  "mcpServers": {
    "elevenlabs": {
      "command": "uv",
      "args": [
        "--directory",
        "d:/GitHub/elevenlabs-mcp-server/src/elevenlabs_mcp",
        "run",
        "elevenlabs-mcp"
      ],
      "env": {
        "ELEVENLABS_API_KEY": "your-api-key",
        "ELEVENLABS_VOICE_ID": "your-voice-id",
        "ELEVENLABS_MODEL_ID": "eleven_flash_v2"
      }
    }
  }
}
```

## Usage

1. Start the server:
   ```bash
   uv --directory d:/GitHub/elevenlabs-mcp-server/src/elevenlabs_mcp run elevenlabs-mcp
   ```

2. Use with any MCP client (e.g., Claude Desktop)

### Tools

- `generate_audio`: Generate audio from a story script and return the audio content directly. The script can include multiple parts with optional voice_id and actor assignments.

## Development

Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## License

MIT
