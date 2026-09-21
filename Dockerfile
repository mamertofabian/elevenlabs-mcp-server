FROM ghcr.io/astral-sh/uv:0.10.11 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install --no-install-recommends --yes ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable && useradd --uid 10001 --create-home voiceover && mkdir /data && chown voiceover:voiceover /data
ENV PATH="/app/.venv/bin:$PATH" ELEVENLABS_OUTPUT_DIR=/data/output ELEVENLABS_DATABASE_PATH=/data/voiceover_history.db PYTHONDONTWRITEBYTECODE=1
USER voiceover
WORKDIR /data
ENTRYPOINT ["elevenlabs-mcp-server"]
