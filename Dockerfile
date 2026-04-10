# =============================================================================
# AI:OS — Agentic Intelligence Operating System
# Multi-stage Dockerfile for production deployment
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies in a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native extensions (numpy, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Joshua Hendricks Cole <admin@aios.is>"
LABEL org.opencontainers.image.title="AI:OS"
LABEL org.opencontainers.image.description="Agentic Intelligence Operating System"
LABEL org.opencontainers.image.source="https://github.com/Workofarttattoo/AioS"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Non-root user for security
RUN groupadd -r aios && useradd -r -g aios -d /app -s /sbin/nologin aios

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Copy application source
COPY aios_cli           ./aios_cli
COPY aios/              ./aios/
COPY agents/            ./agents/
COPY config.py          .
COPY diagnostics.py     .
COPY model.py           .
COPY settings.py        .
COPY apps.py            .
COPY __init__.py        .
COPY tools/__init__.py  ./tools/__init__.py
COPY tools/_toolkit.py  ./tools/_toolkit.py
COPY tools/_stubs.py    ./tools/_stubs.py
COPY gui/               ./gui/
COPY modules/           ./modules/
COPY web/               ./web/
COPY examples/          ./examples/
COPY templates/         ./templates/

# Copy non-encrypted root modules (encrypted ones are skipped gracefully)
COPY *.py               ./

# Health check — tool launcher HTTP endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${AIOS_LAUNCHER_PORT:-7777}/health')" || exit 1

# Drop to non-root
RUN chown -R aios:aios /app
USER aios

# Default: run the CLI in boot mode; override with docker-compose command
EXPOSE 7777
ENTRYPOINT ["python", "aios_cli"]
CMD ["boot"]
