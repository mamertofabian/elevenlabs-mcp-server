"""Distribution-backed version reporting."""

from importlib import metadata


_DISTRIBUTION_NAME = "elevenlabs-mcp-server"
_FALLBACK_VERSION = "0.1.1"


def package_version() -> str:
    """Return installed distribution version or the packaged source fallback."""

    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return _FALLBACK_VERSION
