# Revival implementation and gap review

Date: 2026-09-21. Development build: `0.2.0.dev0`. Not published.

## Outcome

The primary stdio entry point now runs the connected revival workflow:

`plan → submit → durable worker → source publication/verification → interruption → restart → explicit resume with reuse → final assembly → artifact retrieval`

This is exercised through actual MCP stdio processes, the official ElevenLabs SDK with an isolated fake HTTP transport, real SQLite, real MP3 decoding, and actual WAV/MP3 assembly. The process is killed while its fourth request is in flight. After restart, no provider request occurs until explicit resume; the first three verified chunks are reused. TTS and native dialogue both pass this scenario. No paid ElevenLabs requests were made.

## Implemented surfaces

| Surface | Implementation |
|---|---|
| Provider | Official SDK 2.68; TTS and native dialogue; explicit profiles; finite network timeouts; bounded source bytes; hidden synthesis retries disabled; sanitized definite/uncertain failures; available request IDs retained. |
| Worker | Tracked single-owner task, durable plans/reservations/dispatch/results, sequential execution, cumulative ceilings, credential-context binding, cancellation, and orderly shutdown. |
| Recovery | OS ownership guards; no automatic startup synthesis; complete-marker adoption; integrity revalidation of previously decoded successes; explicit uncertainty acknowledgment; assembly-only repair without additional synthesis. |
| Audio | Exact source publication, checksum/fingerprint checks, real decode validation, ordered PCM assembly with explicit pauses, one final encoding, MP3/WAV outputs, and production JSON records. |
| MCP | SDK v2 public callbacks; nine new tools plus six legacy tools; structured envelopes and text fallback; public response-schema validation; artifact resources and bounded inline delivery. |
| Compatibility | Actual SDK-v1 client negotiation, original legacy success/resource envelopes, original actor/voice history values, history playback paths, and safely bounded historical-file retrieval/deletion. Production legacy generation shares the durable core. |
| Storage | Atomic idempotency and mutations; source/final artifact state; online backup of populated legacy databases before additive migration; existing legacy rows preserved; tombstones prevent resubmission after deletion. |
| Delivery | Locked package, installed-wheel smoke tests, non-root CLI-only Docker image, FFmpeg setup, documentation, runtime lint/types, and credential-free CI. |

## Verification evidence

- Full Python suite: **309 passed**. Non-blocking upstream pydub and deprecated compatibility-progress warnings remain.
- `uv run ruff check .`: passed.
- `uv run pyright`: zero errors/warnings.
- `uv build`: wheel and source distribution built for `0.2.0.dev0`.
- Actionlint 1.7.7: both GitHub workflows passed.
- Installed wheel: supported SDK-v2 floor/current resolution; discovery outside the checkout without provider credentials or FFmpeg on PATH.
- Separate real MCP SDK-v1 client: discovery, native planning, and legacy history resource compatibility passed against the SDK-v2 server.
- Container: built and started twice with a read-only root filesystem, non-root UID, isolated network, `/tmp` tmpfs, and a persisted data volume. Planning/discovery worked; uncredentialed submission was refused without creating a job. FFmpeg encode/decode preflight and persisted SQLite were verified. The temporary smoke-test volume was removed.
- Audio ordering: synthetic tones verify output order and exact inserted silence, beyond file-existence checks.

Key reproduction commands:

```bash
uv sync --frozen --dev
uv run pytest -q
uv run ruff check .
uv run pyright
uv build
uv run pytest -q tests/integration/test_revival_mcp.py tests/integration/test_revival_failures.py
```

## Gaps found and fixed

Independent read-only review ran multiple repair loops. Its final runtime verdict was **Approved**, with no remaining actionable findings in the reviewed scope. Findings and associated regressions covered:

- Replacement finals retained corrupt predecessors: old final/production artifacts are now retired atomically.
- Completed jobs could not repair final corruption without restarting: fresh resume verifies and repairs locally without buying audio again.
- Deletion missed orphan/temp files: bounded, no-follow cleanup removes the full owned job tree and supports retries.
- Historical legacy deletion regressed: the safe compatibility path is restored.
- An older cancellation could overwrite a newer resume: mutation orchestration is serialized, with revisions preserved.
- Legacy paths could target the database or unrelated files: generated filenames, ownership, protected-file identity, and alias checks now apply.
- Missing FFmpeg invalidated good completed work: previously decoded bytes are revalidated by integrity, not reclassified as corrupt when a decoder is unavailable.

Additional implementation checks corrected resource discovery, SDK voice-ID route traversal, wire-schema differences, actor-label loss in legacy projection, credential-context creation atomicity, cursor stability across deletion, preservation of uncertain evidence, and active script-payload retention after deletion. Receipt hashes and audit counters remain; deletion is not forensic erasure of SQLite pages or backups.

## Remaining validation boundaries

The next operator test can use the real revival tools with permitted ElevenLabs voices and an explicitly approved spend ceiling. Live account entitlement, network behavior, voice quality, and listening quality have not been claimed from offline evidence.

The secure file and lock implementation currently supports POSIX hosts. Windows users need WSL or the Linux container. The optional SvelteKit console has not been redesigned; public HTTP/multi-user hosting remains outside the core revival scope.

The owner explicitly requested this work outside MAID. Historical manifests were not updated and are not current implementation evidence. CI now verifies actual tests, runtime lint, types, workflows, and packaging. This decision does not silently waive MAID for unrelated future tasks.

No release, push, or implementation commit has been performed for this development increment.
