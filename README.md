# ElevenLabs MCP Server
[![smithery badge](https://smithery.ai/badge/elevenlabs-mcp-server)](https://smithery.ai/server/elevenlabs-mcp-server)

A Model Context Protocol (MCP) server that integrates with ElevenLabs text-to-speech API, featuring both a server component and a sample web-based MCP Client (SvelteKit) for managing voice generation tasks.

<a href="https://glama.ai/mcp/servers/leukzvus7o"><img width="380" height="200" src="https://glama.ai/mcp/servers/leukzvus7o/badge" alt="ElevenLabs Server MCP server" /></a>

## Features

- Generate audio from text using ElevenLabs API
- Support for multiple voices and script parts
- SQLite database for persistent history storage
- Sample SvelteKit MCP Client for:
  - Simple text-to-speech conversion
  - Multi-part script management
  - Voice history tracking and playback
  - Audio file downloads

## Installation

The revival development build is `0.2.0.dev0` and has not been published. Use
Development Installation to test this checkout; registry/`uvx` installation may
still resolve the older published release.

### Installing via Smithery

To install ElevenLabs MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/elevenlabs-mcp-server):

```bash
npx -y @smithery/cli install elevenlabs-mcp-server --client claude
```

### Published release using uvx

When using [`uvx`](https://docs.astral.sh/uv/guides/tools/), no specific installation is needed.

Add the following configuration to your MCP settings file (e.g., `cline_mcp_settings.json` for Claude Desktop):

```json
{
  "mcpServers": {
    "elevenlabs": {
      "command": "uvx",
      "args": ["elevenlabs-mcp-server"],
      "env": {
        "ELEVENLABS_API_KEY": "your-api-key",
        "ELEVENLABS_VOICE_ID": "your-voice-id",
        "ELEVENLABS_MODEL_ID": "eleven_flash_v2",
        "ELEVENLABS_STABILITY": "0.5",
        "ELEVENLABS_SIMILARITY_BOOST": "0.75",
        "ELEVENLABS_STYLE": "0.1",
        "ELEVENLABS_OUTPUT_DIR": "output"
      }
    }
  }
}
```

### Development Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   uv sync --frozen --dev
   ```
3. Copy `.env.example` to `.env` and fill in your ElevenLabs credentials

```json
{
  "mcpServers": {
    "elevenlabs": {
      "command": "uv",
      "args": [
        "--directory",
        "path/to/elevenlabs-mcp-server",
        "run",
        "elevenlabs-mcp-server"
      ],
      "env": {
        "ELEVENLABS_API_KEY": "your-api-key",
        "ELEVENLABS_VOICE_ID": "your-voice-id",
        "ELEVENLABS_MODEL_ID": "eleven_flash_v2",
        "ELEVENLABS_STABILITY": "0.5",
        "ELEVENLABS_SIMILARITY_BOOST": "0.75",
        "ELEVENLABS_STYLE": "0.1",
        "ELEVENLABS_OUTPUT_DIR": "output"
      }
    }
  }
}
```

## Using the Sample SvelteKit MCP Client

1. Navigate to the web UI directory:
   ```bash
   cd clients/web-ui
   ```
2. Install dependencies:
   ```bash
   pnpm install
   ```
3. Copy `.env.example` to `.env` and configure as needed
4. Run the web UI:
   ```bash
   pnpm dev
   ```
5. Open http://localhost:5174 in your browser

### Available Tools

- `generate_audio_simple`: Generate audio from plain text using default voice settings
- `generate_audio_script`: Generate audio from a structured script with multiple voices and actors
- `delete_job`: Delete a job by its ID
- `get_audio_file`: Get the audio file by its ID
- `list_voices`: List all available voices
- `get_voiceover_history`: Get voiceover job history. Optionally specify a job ID for a specific job.

### Available Resources

- `voiceover://history/{job_id}`: Get the audio file by its ID
- `voiceover://voices`: List all available voices

## Recovery candidate status

The revival now connects typed planning, the official ElevenLabs SDK, durable
execution, recovery, verified chunk reuse, and final MP3/WAV assembly through
MCP SDK v2. WAV export is decoded PCM from MP3 source, not original lossless synthesis.
The six legacy generation/history tools remain available; production
legacy generation uses the same durable core. The legacy generation tools remain synchronous
from the client's perspective. Long renders should use submit/poll instead.

Scripts and job history stay local; speech text is sent to ElevenLabs during
explicitly authorized rendering. Jobs run only for the owning server process lifetime.
A second owner of the database directory is rejected. An upstream outcome may be unknown
after a timeout or interruption: restart pauses work and never resynthesizes automatically.
Explicit resume reuses verified completed chunks. Repeating a submit or mutation key
with the same input does not authorize new work; changed input conflicts.

### Setup and validation

Install FFmpeg (including MP3 encoding/decoding) and Python 3.11 or newer.
Default verification limits are 256 MiB of source bytes per attempt, 10 minutes
per decoded chunk, and two hours of final audio including explicit pauses. Plans
whose pauses alone reach the final limit are rejected before any generation. The secure
filesystem and ownership implementation currently supports POSIX hosts; unsupported
platforms fail explicitly. On Windows, use the Linux container or WSL.

```bash
uv sync --frozen --dev
uv run pytest -q
uv run ruff check src
uv run pyright
uv build
```

Revival development intentionally does not use MAID. Historical manifests remain
as historical records; current CI checks tests, runtime lint, types, workflows,
and packaging. The installed historical commit hook may be skipped explicitly for
this revival work with `SKIP=maid-verify`; it is not evidence for the current implementation.

### New tools and an end-to-end render

1. Call `plan_voiceover` with a typed script and explicit options. This needs no key,
   performs no provider calls, and returns the plan hash, chunks, and exposure estimate.
2. Call `submit_voiceover` with the **same** script/options/hash, an idempotency key,
   and cumulative `budget.max_total_characters` / `budget.max_total_requests`.
3. Poll `get_job`. `cancel_voiceover` requires `expected_revision` and an operation key;
   its acknowledgment means scheduling is cancelled, not that upstream billing stopped.
4. After restarting, inspect the paused job. `resume_voiceover` requires the current
   revision, an operation key, and cumulative ceilings. Set `retry_uncertain=true`
   only to authorize possible duplicate charges for unknown/corrupt chunks.
5. Retrieve a final ID using `get_artifact` (`metadata`, `inline`, or `file`). Inline
   data is capped at 8 MiB. A production JSON artifact records ordered source hashes,
   chunk/attempt references, pauses, options, and final output details.

Example planning input (replace both voice IDs with permitted provider voices):

```json
{
  "script": {
    "script_version": "1",
    "cast": {"narrator": {"voice_id": "YOUR_VOICE_A"}, "guest": {"voice_id": "YOUR_VOICE_B"}},
    "scenes": [{"id": "intro", "parts": [
      {"id": "p1", "actor": "narrator", "text": "Welcome to our demonstration.", "pause_after_ms": 300},
      {"id": "p2", "actor": "guest", "text": "We can resume without repeating verified work."}
    ]}]
  },
  "options": {"engine": "tts", "model_id": "eleven_multilingual_v2", "export_format": "mp3"}
}
```

`engine="dialogue", model_id="eleven_v3"` uses native dialogue; it never silently falls
back to TTS. TTS supports multilingual v2, Flash v2/v2.5 and Turbo v2.5.
`search_voices` and `list_models` fetch metadata explicitly. Planning never infers account
entitlement. Generation retries are not hidden inside the SDK: a new attempt requires
an explicit resume within the approved cumulative ceiling.

Credentials use `ELEVENLABS_API_KEY`. `ELEVENLABS_OUTPUT_DIR` and
`ELEVENLABS_DATABASE_PATH` may be separate absolute paths. Only the launch directory's
`.env` is loaded; process environment values take precedence. Discovery works without a key
or FFmpeg. Rendering checks local codec prerequisites before consuming provider credits.
Changing credential context blocks further synthesis for existing jobs; local assembly-only
repair remains possible without provider access.

Before additive migration of a populated legacy database, an online SQLite backup is
written beside it as `*.pre-revival.*.bak` (including committed WAL data). Legacy tables
and IDs remain available through the history tools; old recordings do not gain fabricated
chunk resumability. Historical file reads/deletes must remain inside the configured output
root, with generated job-ID/timestamp filenames and unambiguous ownership. Database
files, unrelated filenames, and aliased paths are rejected. Rollback uses that backup with the old package, never a live schema downgrade.
Deleting native jobs purges active stored script payloads and tombstones receipts
before removing their owned tree, including orphaned files. Audit IDs, hashes and
exposure counters remain; backups and forensic SQLite erasure are outside this operation. Failed cleanup is explicit and can be retried with the same job ID.

See [implementation status and gap-review evidence](docs/revival-implementation-status.md).

### Offline revival acceptance test

```bash
uv run pytest -q tests/integration/test_revival_mcp.py tests/integration/test_revival_failures.py
```

The stdio test uses the real MCP server, official SDK with an isolated fake HTTP transport,
SQLite, actual MP3 decoding, and WAV assembly. It kills the process during the fourth
request, restarts it, verifies no automatic requests, resumes with explicit acknowledgment,
and proves the first three chunks were reused. No ElevenLabs credits or credentials are
used. This is not production readiness or live-provider/audio-quality evidence: the latter
requires separately authorized voices and spend.

### Container

```bash
docker build -t elevenlabs-revival:local .
# Optional offline container acceptance check:
uv run python scripts/smoke_container.py
docker volume create elevenlabs-data
docker run --rm -i --read-only --tmpfs /tmp -v elevenlabs-data:/data \
  -e ELEVENLABS_API_KEY elevenlabs-revival:local
```

The image is stdio-only, runs as a non-root user, installs the actual locked package,
and requires persisted `/data`. It exposes no HTTP port. The existing SvelteKit console
remains optional; it has not been redesigned for the new asynchronous tools.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
