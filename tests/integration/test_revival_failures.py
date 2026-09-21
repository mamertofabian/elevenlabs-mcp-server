from __future__ import annotations

import asyncio
import threading

import pytest

from elevenlabs_mcp.jobs import VoiceoverService
from tests.integration.test_revival_end_to_end import (
    OPTIONS,
    SCRIPT,
    Provider,
    settled,
)


async def submit(service, key="submit", requests=10):
    plan = service.plan(SCRIPT, OPTIONS)
    return await service.submit(
        SCRIPT,
        OPTIONS,
        plan["plan_hash"],
        key,
        {"max_total_characters": 1000, "max_total_requests": requests},
    )


@pytest.mark.asyncio
async def test_cancel_stops_scheduling_and_old_receipt_cannot_cancel_resumed_work(
    tmp_path, audio
):
    entered = threading.Event()
    release = threading.Event()

    class Slow(Provider):
        def generate(self, request, options):
            self.calls.append(request.chunk_id)
            entered.set()
            release.wait(5)
            yield self.audio

    provider = Slow(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        job = await submit(service)
        assert await asyncio.to_thread(entered.wait, 3)
        ack = await service.cancel(job["job_id"], 0, "cancel")
        assert ack["cancel_requested"] and ack["status"] == "running"
        release.set()
        cancelled = await settled(service, job["job_id"])
        assert cancelled["status"] == "cancelled" and len(provider.calls) == 1
        await service.resume(
            job["job_id"],
            1,
            "resume",
            {"max_total_characters": 1000, "max_total_requests": 10},
        )
        await service.cancel(job["job_id"], 0, "cancel")
        done = await settled(service, job["job_id"])
        assert done["status"] == "completed" and len(provider.calls) == 5
    finally:
        release.set()
        await service.close()


@pytest.mark.asyncio
async def test_assembly_only_retry_needs_no_provider_and_no_new_reservations(
    tmp_path, audio, monkeypatch
):
    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    original = service.assembler.assemble

    def broken(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(service.assembler, "assemble", broken)
    try:
        job = await submit(service)
        failed = await settled(service, job["job_id"])
        assert failed["reason"] == "ASSEMBLY_FAILED" and failed["verified_chunks"] == 5
        monkeypatch.setattr(service.assembler, "assemble", original)
        monkeypatch.setattr(
            provider,
            "check_ready",
            lambda _: (_ for _ in ()).throw(
                AssertionError("No provider needed for assembly")
            ),
        )
        await service.resume(
            job["job_id"],
            0,
            "resume",
            {"max_total_characters": 1000, "max_total_requests": 5},
        )
        done = await settled(service, job["job_id"])
        assert (
            done["status"] == "completed"
            and done["reserved_requests"] == 5
            and len(provider.calls) == 5
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_corrupt_chunk_requires_explicit_recovery_and_regenerates_only_that_chunk(
    tmp_path, audio
):
    provider = Provider(audio, 4)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        job = await submit(service)
        await settled(service, job["job_id"])
        chunks = await service.store.chunks(job["job_id"])
        (service.root / chunks[0]["relative_path"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="ARTIFACT_CORRUPT"):
            await service.resume(
                job["job_id"],
                0,
                "resume",
                {"max_total_characters": 1000, "max_total_requests": 10},
            )
        await service.resume(
            job["job_id"],
            0,
            "resume",
            {"max_total_characters": 1000, "max_total_requests": 10},
            True,
        )
        done = await settled(service, job["job_id"])
        assert done["status"] == "completed" and done["reserved_requests"] == 7
        assert provider.calls.count(chunks[0]["chunk_id"]) == 2
        assert provider.calls.count(chunks[1]["chunk_id"]) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_budget_pause_and_increased_ceiling_do_not_replay_verified_audio(
    tmp_path, audio
):
    provider = Provider(audio, 4)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        job = await submit(service, requests=5)
        await settled(service, job["job_id"])
        await service.resume(
            job["job_id"],
            0,
            "resume",
            {"max_total_characters": 1000, "max_total_requests": 5},
            True,
        )
        paused = await settled(service, job["job_id"])
        assert paused["reason"] == "BUDGET_EXCEEDED" and len(provider.calls) == 5
        await service.resume(
            job["job_id"],
            1,
            "more",
            {"max_total_characters": 1000, "max_total_requests": 6},
        )
        done = await settled(service, job["job_id"])
        assert done["status"] == "completed" and len(provider.calls) == 6
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_delete_tombstone_prevents_idempotent_resubmission(tmp_path, audio):
    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        job = await submit(service)
        done = await settled(service, job["job_id"])
        await service.delete(job["job_id"])
        await service.delete(job["job_id"])
        async with (
            service.store.connect() as db,
            db.execute(
                "SELECT plan_json,normalized_script_json FROM voiceover_jobs WHERE job_id=?",
                (job["job_id"],),
            ) as cursor,
        ):
            deleted_row = await cursor.fetchone()
            assert deleted_row is not None
            assert tuple(deleted_row) == ("{}", "{}")
        from elevenlabs_mcp.database import IdempotencyConflictError

        with pytest.raises(IdempotencyConflictError):
            await submit(service, requests=11)
        with pytest.raises(ValueError, match="JOB_DELETED"):
            await submit(service)
        with pytest.raises(ValueError, match="ARTIFACT_NOT_FOUND"):
            await service.get_artifact(done["final_artifact_ids"][0])
        assert len(provider.calls) == 5
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_final_repair_retires_corrupt_outputs_and_survives_next_restart(
    tmp_path, audio
):
    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    job = await submit(service)
    done = await settled(service, job["job_id"])
    old_id = done["final_artifact_ids"][0]
    record = await service.store.artifact(old_id)
    (service.root / record["relative_path"]).write_bytes(b"corrupt")
    try:
        await service.resume(
            job["job_id"],
            0,
            "repair",
            {"max_total_characters": 1000, "max_total_requests": 5},
        )
        repaired = await settled(service, job["job_id"])
        assert repaired["status"] == "completed"
        assert (
            len(repaired["final_artifact_ids"]) == 1
            and old_id not in repaired["final_artifact_ids"]
        )
        with pytest.raises(ValueError, match="ARTIFACT_NOT_FOUND"):
            await service.get_artifact(old_id)
        assert len(provider.calls) == 5
    finally:
        await service.close()
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        assert (await service.get_job(job["job_id"]))["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_delete_removes_orphans_but_never_follows_external_symlinks(
    tmp_path, audio
):
    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        job = await submit(service)
        await settled(service, job["job_id"])
        directory = service.root / "jobs" / job["job_id"]
        (directory / "final" / "orphan.mp3").write_bytes(b"private orphan")
        outside = tmp_path / "external"
        outside.mkdir()
        (outside / "private").write_bytes(b"untouched")
        (directory / "escape").symlink_to(outside, target_is_directory=True)
        await service.delete(job["job_id"])
        assert not directory.exists()
        assert (outside / "private").read_bytes() == b"untouched"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_legacy_history_retrieval_and_safe_delete_remain_available(
    tmp_path, audio
):
    import json

    from elevenlabs_mcp.config import Settings
    from elevenlabs_mcp.mcp_app import RevivalTools
    from elevenlabs_mcp.models import AudioJob
    from elevenlabs_mcp.server import ElevenLabsServer

    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=tmp_path / "out",
        database_path=tmp_path / "state.db",
        database_path_explicit=True,
    )
    service = VoiceoverService(
        settings.database_path, settings.output_dir, Provider(audio)
    )
    server = ElevenLabsServer(
        settings, environ={"ELEVENLABS_API_KEY": ""}, revival_service=service
    )
    await server.initialize()
    path = settings.output_dir / "full_audio_historical.mp3"
    path.write_bytes(audio)
    job = AudioJob.create(
        id="historical",
        status="completed",
        script_parts=[{"text": "old"}],
        total_parts=1,
    )
    job.output_file = str(path)
    await server.db.insert_job(job)
    tools = RevivalTools(service, server)
    try:
        result = await tools.call("get_voiceover_history", {"job_id": "historical"})
        assert json.loads(result.content[0].text)[0]["id"] == "historical"
        result = await tools.call("get_audio_file", {"job_id": "historical"})
        assert not result.is_error
        result = await tools.call("delete_job", {"job_id": "historical"})
        assert not result.is_error and not path.exists()
        assert await server.db.get_job("historical") is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_upgrade_preserves_legacy_rows_and_online_backup_includes_wal(
    tmp_path, audio
):
    import sqlite3

    from elevenlabs_mcp.database import Database
    from elevenlabs_mcp.models import AudioJob

    path = tmp_path / "state.db"
    legacy = Database(path)
    await legacy.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        job = AudioJob.create(
            id="old",
            status="completed",
            script_parts=[{"text": "preserved"}],
            total_parts=1,
        )
        await legacy.insert_job(job)
        service = VoiceoverService(path, tmp_path / "out", Provider(audio))
        await service.start()
        try:
            backups = list(tmp_path.glob("*.pre-revival.*.bak"))
            assert len(backups) == 1
            with sqlite3.connect(backups[0]) as backup:
                assert backup.execute(
                    "SELECT id,script_parts FROM audio_jobs"
                ).fetchone() == ("old", '[{"text": "preserved"}]')
            preserved = await legacy.get_job("old")
            assert preserved is not None
            assert preserved.script_parts == [{"text": "preserved"}]
        finally:
            await service.close()


@pytest.mark.asyncio
async def test_changed_credentials_cannot_resume_generation(tmp_path, audio):
    provider = Provider(audio, 4)
    provider.context_id = "account-a"
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    job = await submit(service)
    await settled(service, job["job_id"])
    await service.close()
    replacement = Provider(audio)
    replacement.context_id = "account-b"
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", replacement)
    await service.start()
    try:
        with pytest.raises(ValueError, match="CREDENTIAL_CONTEXT_CHANGED"):
            await service.resume(
                job["job_id"],
                0,
                "resume",
                {"max_total_characters": 1000, "max_total_requests": 10},
                True,
            )
        assert replacement.calls == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_initial_underbudget_plan_never_dispatches_or_persists(tmp_path, audio):
    from elevenlabs_mcp.database import BudgetExceededError

    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        with pytest.raises(BudgetExceededError):
            await submit(service, requests=4)
        assert provider.calls == [] and await service.store.ids() == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pagination_cursor_survives_deleting_its_last_job(tmp_path, audio):
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    try:
        plan = service._make_plan(SCRIPT, OPTIONS)
        ids = []
        for i in range(3):
            identity = str(uuid4())
            ids.append(identity)
            await service.store.create_job(
                "local",
                identity,
                plan,
                str(i),
                1000,
                10,
                datetime.now(UTC) + timedelta(seconds=i),
            )
            await service.store.transition(
                identity, "cancelled", None, datetime.now(UTC)
            )
        page = await service.list_jobs(limit=2)
        assert [j["job_id"] for j in page["jobs"]] == list(reversed(ids[1:]))
        await service.delete(ids[1])
        following = await service.list_jobs(limit=2, cursor=page["next_cursor"])
        assert [j["job_id"] for j in following["jobs"]] == ids[:1]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cancel_cannot_overwrite_newer_resume(tmp_path, audio, monkeypatch):
    provider = Provider(audio, 4)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    job = await submit(service)
    await settled(service, job["job_id"])
    entered = asyncio.Event()
    release = asyncio.Event()
    transition = service.store.transition

    async def delayed(job_id, status, reason, now):
        if status == "cancelled":
            entered.set()
            await release.wait()
        return await transition(job_id, status, reason, now)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(service.store, "transition", delayed)
            patch.setattr(service._wake, "set", lambda: None)
            patch.setattr(service.assembler, "check_dependencies", lambda: None)
            cancel = asyncio.create_task(service.cancel(job["job_id"], 0, "cancel"))
            await asyncio.wait_for(entered.wait(), 3)
            resume = asyncio.create_task(
                service.resume(
                    job["job_id"],
                    1,
                    "resume",
                    {"max_total_characters": 1000, "max_total_requests": 10},
                    True,
                )
            )
            try:
                await asyncio.wait_for(asyncio.shield(resume), 0.25)
            except TimeoutError:
                pass
            release.set()
            await cancel
            await resume
            state = await service.get_job(job["job_id"])
            assert (state["status"], state["revision"], state["cancel_requested"]) == (
                "queued",
                2,
                False,
            )
    finally:
        release.set()
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["database", "unrelated", "other_job", "database_hardlink"]
)
async def test_tampered_legacy_record_cannot_read_or_delete_protected_files(
    tmp_path, audio, target
):
    import os

    from elevenlabs_mcp.config import Settings
    from elevenlabs_mcp.mcp_app import RevivalTools
    from elevenlabs_mcp.models import AudioJob
    from elevenlabs_mcp.server import ElevenLabsServer

    root = tmp_path / "data"
    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=root,
        database_path=root / "voiceover_history.db",
        database_path_explicit=True,
    )
    service = VoiceoverService(settings.database_path, root, Provider(audio))
    server = ElevenLabsServer(
        settings, environ={"ELEVENLABS_API_KEY": ""}, revival_service=service
    )
    await server.initialize()
    if target == "database":
        path = settings.database_path
    elif target == "database_hardlink":
        path = root / "full_audio_historical.mp3"
        os.link(settings.database_path, path)
    else:
        path = root / (
            "unrelated.mp3" if target == "unrelated" else "full_audio_other-job.mp3"
        )
        path.write_bytes(audio)
    job = AudioJob.create(
        id="historical",
        status="completed",
        script_parts=[{"text": "old"}],
        total_parts=1,
    )
    job.output_file = str(path)
    await server.db.insert_job(job)
    tools = RevivalTools(service, server)
    try:
        assert (await tools.call("get_audio_file", {"job_id": "historical"})).is_error
        assert (await tools.call("delete_job", {"job_id": "historical"})).is_error
        assert path.exists() and settings.database_path.exists()
        assert await server.db.get_job("historical") is not None
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_resume_adopts_already_received_audio_after_local_decode_failure(
    tmp_path, audio, monkeypatch, cancel_first
):
    from elevenlabs_mcp.audio import AudioVerificationError

    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    verify = service.verifier.verify

    def failure(*args):
        raise AudioVerificationError("temporary decoder failure")

    monkeypatch.setattr(service.verifier, "verify", failure)
    try:
        job = await submit(service)
        paused = await settled(service, job["job_id"])
        assert paused["status"] == "paused" and len(provider.calls) == 1
        if cancel_first:
            await service.cancel(job["job_id"], 0, "cancel")
        monkeypatch.setattr(service.verifier, "verify", verify)
        await service.resume(
            job["job_id"],
            1 if cancel_first else 0,
            "resume",
            {"max_total_characters": 1000, "max_total_requests": 5},
        )
        done = await settled(service, job["job_id"])
        assert done["status"] == "completed" and len(provider.calls) == 5
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_missing_ffmpeg_preserves_completed_work_and_noop_resume(tmp_path, audio):
    provider = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    await service.start()
    job = await submit(service)
    done = await settled(service, job["job_id"])
    await service.close()
    replacement = Provider(audio)
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", replacement)
    from elevenlabs_mcp.audio import FFmpegDecoder

    service.verifier.decoder = FFmpegDecoder("missing-review-ffmpeg")
    service.assembler.decoder.executable = "missing-review-ffmpeg"
    await service.start()
    try:
        current = await service.get_job(job["job_id"])
        assert current["status"] == "completed" and current["verified_chunks"] == 5
        resumed = await service.resume(
            job["job_id"],
            0,
            "noop",
            {"max_total_characters": 1000, "max_total_requests": 5},
        )
        assert resumed["status"] == "completed" and replacement.calls == []
        assert (await service.get_artifact(done["final_artifact_ids"][0], "inline"))[
            "data"
        ].startswith(b"RIFF")
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_new_state_is_private_and_future_runtime_schema_is_rejected(tmp_path):
    import sqlite3
    import stat

    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", Provider(b""))
    await service.start()
    assert stat.S_IMODE((tmp_path / "state.db").stat().st_mode) == 0o600
    await service.close()
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("DROP TABLE runtime_schema")
        db.execute("CREATE TABLE runtime_schema(version INTEGER PRIMARY KEY)")
        db.execute("INSERT INTO runtime_schema VALUES(2)")
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", Provider(b""))
    with pytest.raises(RuntimeError, match="Unsupported runtime schema"):
        await service.start()
    with sqlite3.connect(tmp_path / "state.db") as db:
        assert db.execute("SELECT version FROM runtime_schema").fetchall() == [(2,)]


@pytest.mark.asyncio
async def test_shared_legacy_generation_preserves_original_actor_metadata(
    tmp_path, audio
):
    import json

    from elevenlabs_mcp.config import Settings
    from elevenlabs_mcp.mcp_app import RevivalTools
    from elevenlabs_mcp.server import ElevenLabsServer

    settings = Settings(
        launch_cwd=tmp_path,
        output_dir=tmp_path / "out",
        database_path=tmp_path / "state.db",
        database_path_explicit=True,
    )
    service = VoiceoverService(
        settings.database_path, settings.output_dir, Provider(audio)
    )
    server = ElevenLabsServer(
        settings, environ={"ELEVENLABS_API_KEY": ""}, revival_service=service
    )
    await server.initialize()
    tools = RevivalTools(service, server)
    parts = [{"text": "Hello.", "actor": "Dr. Example", "voice_id": "custom-voice"}]
    try:
        rendered = await tools.call(
            "generate_audio_script", {"script": json.dumps(parts)}
        )
        assert (
            not rendered.is_error
            and rendered.content[0].text.splitlines()[0]
            == "Audio generation successful."
        )
        history = await tools._history()
        assert len(history) == 1 and history[0]["script_parts"] == parts
        assert history[0]["output_file"]
    finally:
        await service.close()


def test_known_overlong_pause_plan_is_rejected_before_any_work(tmp_path):
    from elevenlabs_mcp.planner import PlanningLimitError

    provider = Provider(b"")
    service = VoiceoverService(tmp_path / "state.db", tmp_path / "out", provider)
    script = {
        "script_version": "1",
        "cast": {"a": {"voice_id": "voice"}},
        "scenes": [
            {
                "id": "s",
                "parts": [
                    {
                        "id": f"p{i}",
                        "actor": "a",
                        "text": "Hello.",
                        "pause_after_ms": 10000,
                    }
                    for i in range(800)
                ],
            }
        ],
    }
    with pytest.raises(PlanningLimitError):
        service.plan(script, OPTIONS)
    assert provider.calls == [] and not (tmp_path / "state.db").exists()
