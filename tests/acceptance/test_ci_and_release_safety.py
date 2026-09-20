from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[2]


def _read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _yaml(relative: str) -> dict:
    return yaml.load(_read(relative), Loader=yaml.BaseLoader)


def test_ci_runs_credential_free_python_and_maid_gates() -> None:
    assert (PROJECT_ROOT / ".github/workflows/ci.yml").exists()
    workflow = _read(".github/workflows/ci.yml")

    assert "pull_request:" in workflow
    assert re.search(r"\bpush:\s*\n", workflow)
    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert "uv sync --frozen --dev" in workflow
    assert "uv run pytest -q" in workflow
    assert "uv run maid validate" in workflow
    assert "uv run maid test" in workflow
    assert "uv build" in workflow
    assert "docker://rhysd/actionlint:1.7.7" in workflow
    assert "test_installed_wheel_smoke.py" not in workflow
    assert "ELEVENLABS_API_KEY" not in workflow
    assert "pypi" not in workflow.lower()
    document = _yaml(".github/workflows/ci.yml")
    assert set(document["on"]) == {"pull_request", "push"}
    assert document["jobs"]["python"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
    ]


def test_publication_workflow_is_manual_and_confirmed() -> None:
    workflow = _read(".github/workflows/publish-to-test-pypi.yml")
    trigger = workflow.split("jobs:", 1)[0]

    assert "workflow_dispatch:" in trigger
    assert "target:" in trigger
    assert "confirmation:" in trigger
    assert "push:" not in trigger
    assert "pull_request:" not in trigger
    assert "tags:" not in trigger
    assert "branches:" not in trigger
    assert workflow.count("inputs.confirmation == 'publish'") >= 3
    assert "inputs.target == 'pypi'" in workflow
    assert "inputs.target == 'testpypi'" in workflow
    document = _yaml(".github/workflows/publish-to-test-pypi.yml")
    assert set(document["on"]) == {"workflow_dispatch"}
    jobs = document["jobs"]
    assert jobs["publish-to-testpypi"]["if"] == (
        "inputs.confirmation == 'publish' && inputs.target == 'testpypi'"
    )
    production_condition = jobs["publish-to-pypi"]["if"]
    assert "inputs.confirmation == 'publish'" in production_condition
    assert "inputs.target == 'pypi'" in production_condition
    assert "startsWith(github.ref, 'refs/tags/')" in production_condition
    assert jobs["github-release"]["needs"] == "publish-to-pypi"


def test_pytest_asyncio_scope_is_explicit() -> None:
    project = tomllib.loads(_read("pyproject.toml"))

    assert project["tool"]["pytest"]["ini_options"] == {
        "asyncio_default_fixture_loop_scope": "function"
    }


def test_readme_states_recovery_commands_and_limits() -> None:
    readme = _read("README.md")
    assert "## Recovery candidate status" in readme
    section = readme.split("## Recovery candidate status", 1)[1]

    for text in (
        "uv sync --frozen --dev",
        "uv run pytest -q",
        "uv run maid validate",
        "Scripts and job history stay local",
        "speech text is sent to ElevenLabs",
        "legacy generation tools remain synchronous",
        "process lifetime",
        "upstream outcome may be unknown",
        "Durable resume is not implemented yet",
        "not production readiness",
    ):
        assert text in section
