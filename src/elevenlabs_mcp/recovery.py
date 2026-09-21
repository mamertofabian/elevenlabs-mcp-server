"""Local-only recovery of fully published and decoded source artifacts."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict

from .artifacts import ArtifactVerificationError
from .audio import AudioDependencyError, AudioVerificationError, AudioVerifier
from .contracts import RestartReconciliationResult
from .database import AttemptStateError, JobStore
from .workspace import WorkspaceLock


class ArtifactRecoveryFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str
    code: Literal[
        "INTEGRITY_CHECK_FAILED",
        "AUDIO_CHECK_FAILED",
        "STATE_CHANGED",
        "DEPENDENCY_MISSING",
    ]


class ArtifactRecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reconciliation: RestartReconciliationResult
    adopted_artifact_ids: tuple[str, ...]
    failures: tuple[ArtifactRecoveryFailure, ...]


class ArtifactRecovery:
    """Use the configured verifier root, which may differ from the database directory."""

    def __init__(self, store: JobStore, verifier: AudioVerifier) -> None:
        self.store = store
        self.verifier = verifier

    async def recover(
        self, ownership: WorkspaceLock, recovered_at: datetime
    ) -> ArtifactRecoveryResult:
        """Require caller-held ownership for the entire lifecycle; never synthesize."""
        reconciliation = await self.store.reconcile_interrupted(ownership, recovered_at)
        result = await self.adopt_ready(ownership, recovered_at)
        return result.model_copy(update={"reconciliation": reconciliation})

    async def adopt_ready(
        self,
        ownership: WorkspaceLock,
        recovered_at: datetime,
        job_id: str | None = None,
    ) -> ArtifactRecoveryResult:
        """Adopt published results without reconciling or interrupting other live jobs."""
        reconciliation = RestartReconciliationResult(
            paused_job_ids=(), uncertain_attempt_ids=(), abandoned_reservation_ids=()
        )
        candidates = await self.store.recovery_candidates(ownership, job_id)
        adopted: list[str] = []
        failures: list[ArtifactRecoveryFailure] = []
        for identity in candidates:
            try:
                verified = await asyncio.to_thread(self.verifier.verify, identity)
            except ArtifactVerificationError:
                failures.append(
                    ArtifactRecoveryFailure(
                        attempt_id=identity.attempt_id,
                        code="INTEGRITY_CHECK_FAILED",
                    )
                )
                continue
            except AudioDependencyError:
                failures.append(
                    ArtifactRecoveryFailure(
                        attempt_id=identity.attempt_id, code="DEPENDENCY_MISSING"
                    )
                )
                continue
            except AudioVerificationError:
                failures.append(
                    ArtifactRecoveryFailure(
                        attempt_id=identity.attempt_id,
                        code="AUDIO_CHECK_FAILED",
                    )
                )
                continue
            key = json.dumps(
                identity.model_dump(), sort_keys=True, separators=(",", ":")
            )
            artifact_id = str(uuid5(NAMESPACE_URL, key))
            try:
                result = await self.store.record_verified_artifact(
                    artifact_id,
                    verified,
                    recovered_at,
                    recovery_owner=ownership,
                )
            except AttemptStateError:
                failures.append(
                    ArtifactRecoveryFailure(
                        attempt_id=identity.attempt_id,
                        code="STATE_CHANGED",
                    )
                )
                continue
            adopted.append(result.artifact_id)
        return ArtifactRecoveryResult(
            reconciliation=reconciliation,
            adopted_artifact_ids=tuple(adopted),
            failures=tuple(failures),
        )
