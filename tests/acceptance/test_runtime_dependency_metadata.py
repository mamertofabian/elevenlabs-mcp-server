from __future__ import annotations

import email
import subprocess
import tomllib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def _pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _lock() -> dict:
    return tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))


def test_project_runtime_dependencies_bound_mcp_and_exclude_pytest() -> None:
    document = _pyproject()
    project = document["project"]
    runtime = project["dependencies"]

    assert project["name"] == "elevenlabs-mcp-server"
    assert project["license"] == {"file": "LICENSE"}
    assert project["scripts"] == {"elevenlabs-mcp-server": "elevenlabs_mcp.server:main"}
    assert "mcp>=1.1.2,<2" in runtime
    assert "pydantic>=2.10,<3" in runtime
    assert not any(requirement.startswith("pytest") for requirement in runtime)
    assert set(project["optional-dependencies"]["dev"]) == {
        "pytest",
        "pytest-asyncio",
    }
    assert "tool" not in document or "uv" not in document["tool"]
    assert set(document["dependency-groups"]["dev"]) == {
        "maid-runner==2.27.6",
        "pyright>=1.1.389",
        "pytest>=8.3.3",
        "pytest-asyncio",
        "ruff>=0.8.1",
    }


def test_lock_resolves_tested_mcp_v1_without_runtime_pytest() -> None:
    document = _lock()
    packages = {package["name"]: package for package in document["package"]}
    project = packages["elevenlabs-mcp-server"]
    mcp = packages["mcp"]
    requirements = project["metadata"]["requires-dist"]

    assert mcp["version"] == "1.1.2"
    assert {item["name"] for item in project["dependencies"]} == {
        "aiosqlite",
        "mcp",
        "pydub",
        "pydantic",
        "python-dotenv",
        "requests",
        "tenacity",
    }
    mcp_requirement = next(item for item in requirements if item["name"] == "mcp")
    pydantic_requirement = next(
        item for item in requirements if item["name"] == "pydantic"
    )
    assert mcp_requirement["specifier"] == ">=1.1.2,<2"
    assert pydantic_requirement["specifier"] == ">=2.10,<3"
    assert packages["pydantic"]["version"] == "2.10.4"
    assert not any(
        item["name"] == "pytest" and "marker" not in item for item in requirements
    )


def test_built_wheel_preserves_identity_license_and_safe_requirements(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    wheels = list(output_dir.glob("elevenlabs_mcp_server-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        entry_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = email.message_from_bytes(wheel.read(metadata_name))
        entries = wheel.read(entry_name).decode("utf-8")

    requirements = metadata.get_all("Requires-Dist") or []
    mcp_requirement = next(item for item in requirements if item.startswith("mcp"))
    pydantic_requirement = next(
        item for item in requirements if item.startswith("pydantic")
    )
    assert metadata["Name"] == "elevenlabs-mcp-server"
    assert ">=1.1.2" in mcp_requirement
    assert "<2" in mcp_requirement
    assert ">=2.10" in pydantic_requirement
    assert "<3" in pydantic_requirement
    assert not any(
        item.startswith("pytest") and "extra == 'dev'" not in item
        for item in requirements
    )
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert "elevenlabs-mcp-server = elevenlabs_mcp.server:main" in entries
    assert "elevenlabs_mcp/server.py" in names
    assert not any(name.startswith("tests/") for name in names)
