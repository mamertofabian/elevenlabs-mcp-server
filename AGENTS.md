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

<!-- BEGIN MAID RUNNER -->
## MAID Runner

Instruction payload version: 2026.09.18.1

### MAID Codex Skills Workflow
Use the installed MAID Codex skills for manifest-driven development: `maid-planner`, `maid-plan-review`, `maid-implement-draft`, `maid-implementer`, `maid-implementation-review`, `maid-evolver`, `maid-auditor`, `maid-incident-logger`, `maid-outcome-enrich`, `maid-run-review`.

For new features, bug fixes, and refactors, plan with `maid-planner`, review with `maid-plan-review`, implement with `maid-implementer`, and review the result with `maid-implementation-review` before handoff. When continuing from `manifests/drafts/*.manifest.yaml`, use `maid-implement-draft` to harden, lock, promote, implement, review, and capture Outcome.

Before editing a file during an active MAID task, run `maid hook scope-check --path <file>` and treat exit code 2 as out-of-scope. This pre-edit hook check is advisory and does not replace `maid verify` changed-scope validation.

Before treating a file's language as unsupported, run `maid validators` and install a matching validator plugin when available instead of skipping MAID for that file.

Draft manifests under `manifests/drafts/` are planning inventory, not active contracts. Child implementation drafts live at `manifests/drafts/*.manifest.yaml`; epic planning records live at `manifests/drafts/*.epic.yaml` and use split-before-promote before implementation; archived draft records are historical inventory. Before promoting the selected child draft, refresh the Outcome index when needed and run `uv run maid recall --for-manifest manifests/drafts/<slug>.manifest.yaml --plan-packet` when completed Outcome records exist. Recall is advisory planning context only: it can inform draft hardening and implementation risks, but it does not expand scope or replace red evidence, behavioral validation, plan lock, implementation validation, or review. Use `uv run maid insights` to review recurring Outcome lessons when an index is available. To intentionally include instructive failed or abandoned Outcome lessons, refresh the index with `uv run maid learn --include-status completed --include-status abandoned`, then recall from that index; the completed-only default is unchanged. When related Outcome evidence is retrieved, do not dump a raw recall or insights transcript into the task. Digest it visibly: name applicable lessons, reject stale or irrelevant lessons with a reason, and state what changed because of the evidence for the current planning, implementation, or review phase. Recalled, aggregated, and digested Outcomes remain advisory planning context only; they do not create an approval, promotion, done, or review gate. Promote one selected child draft with `uv run maid manifest promote manifests/drafts/<slug>.manifest.yaml`. Do not manually move or copy draft manifests. For metadata-only reference cleanup on locked active manifests, use `uv run maid plan revise <manifest> --reason "<text>" --preserve-red-evidence`. For review-driven behavioral contract changes while implementation is uncommitted, use `uv run maid plan revise <manifest> --reason "<text>" --stash-implementation` so MAID temporarily hides declared implementation changes while it captures fresh red evidence. This cannot hide committed implementation. If revised tests fail for missing behavior in committed code, plain `maid plan revise` can capture fresh red evidence before the fix. If they already pass and valid evidence cannot be preserved, report the evidence blocker and the pre-implementation baseline needed for recovery; do not reset shared history or bypass evidence requirements. See 'Revision Evidence After Implementation' in `docs/draft-manifest-workflow.md`. While a draft or promoted contract belongs to the current unmerged task, revise that same contract in place; use `maid plan revise` when it is locked. A contract is durable when accepted into shared project history, including integration or release branches regardless of name; a local task commit alone is not acceptance. Check repository policy and history, and clarify uncertain acceptance before rewriting. Use new manifests for genuinely separate work or evolution of durable contracts through chain merging or `maid-evolver` supersession.

Declare each touched source file in `files.create`, `files.edit`, `files.scope`, or `files.delete`; `files.read` is context, not writable scope. When public behavior changes in a touched file, add focused behavioral or characterization tests and declare any affected public artifacts in `files.create` or `files.edit`, even if the file was previously listed only in `files.scope`. Use `files.scope` for narrow no-artifact wiring covered by behavioral tests instead of forcing a whole-file public artifact snapshot.

Always capture an Outcome record after implementation validation and implementation review, before final handoff. Capture Outcome after implementation review so the result records the reviewed evidence. Outcome capture is required for completed, partial, failed, superseded, archived, or abandoned MAID work. The Outcome must cite concrete validation evidence and review notes; it does not replace behavioral tests, declared artifacts, validation commands, or implementation review. After Outcome capture, run `uv run maid learn` to refresh the local `.maid/outcomes.json` advisory index for subsequent recall. `.maid/outcomes.json` is generated and ignored; do not commit it. If `maid learn` fails, report the refresh failure as advisory unless recall or insights are required for the current task. See `docs/draft-manifest-workflow.md` and `docs/manifest-outcome-records.md`.

Installed Codex skill-local agent metadata files: 10.
<!-- END MAID RUNNER -->
