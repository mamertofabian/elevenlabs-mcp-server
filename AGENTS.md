# Repository Guidelines

## Project Structure & Module Organization

The Python MCP server lives in `src/elevenlabs_mcp/`. `server.py` defines MCP tools and resources, `elevenlabs_api.py` handles API requests and audio assembly, `database.py` manages SQLite, and `models.py` contains data models. Python tests are in `tests/`.

The sample SvelteKit client is under `clients/web-ui/`. Routes are in `src/routes/`, endpoints in `src/routes/api/`, components in `src/lib/components/`, and the MCP adapter in `src/lib/elevenlabs-client.ts`. Generated audio and SQLite files belong in untracked `output/`.

## Build, Test, and Development Commands

- `uv sync --dev`: install the Python package and development tools from `uv.lock`.
- `uv run elevenlabs-mcp-server`: run the MCP server over stdio.
- `ELEVENLABS_API_KEY=test uv run pytest -q`: run the current Python test suite without making API calls.
- `uv run ruff check .`: lint Python sources and tests.
- `uv run pyright`: perform Python type checking.
- `cd clients/web-ui && pnpm install --frozen-lockfile`: install locked frontend dependencies.
- `pnpm dev`: start the sample client on port 5174.
- `pnpm check && pnpm build`: type-check Svelte code and create a production build.

## Coding Style & Naming Conventions

Use four spaces in Python and conventional PEP 8 naming: `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Add type annotations to public interfaces and keep network, database, and filesystem dependencies injectable where practical.

Use TypeScript for frontend code. Follow existing Svelte conventions: `PascalCase.svelte` components, lowercase route directories, and `+server.ts` for endpoint handlers. Run Ruff, Pyright, and `pnpm check` before opening a PR.

## Testing Guidelines

Pytest is the Python test framework. Name files `test_*.py` and tests `test_<behavior>`. Add focused tests for parsing, MCP responses, API failures, persistence, and partial audio generation. Mock external ElevenLabs calls; tests must not consume credits or require real credentials. Frontend changes should include tests when infrastructure exists and must at least pass `pnpm check` and `pnpm build`.

## Commit & Pull Request Guidelines

History is sparse and mostly uses short imperative subjects. Prefer descriptive imperative commits such as `Fix MCP audio resource response`. For substantial changes, include a body explaining motivation, risks, and verification. PRs should summarize behavior changes, link relevant issues, list validation commands, note configuration changes, and include screenshots for visible UI work.

## Security & Configuration

Keep `.env`, API keys, generated audio, and databases out of Git. Start from `clients/web-ui/.env.example`. Treat web endpoints as local-development examples unless authentication, authorization, and deployment protections are added.
