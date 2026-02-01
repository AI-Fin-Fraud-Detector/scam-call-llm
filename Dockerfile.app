FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1

ARG PROJECT_PATH
WORKDIR ${PROJECT_PATH}

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev

################################################################################

FROM python:3.12-slim-bookworm

ARG PROJECT_PATH
WORKDIR ${PROJECT_PATH}

# Copy the environment
COPY --from=builder ${PROJECT_PATH}/.venv .venv

# Copy the application
COPY app.py .

# Use relative path to uvicorn in the virtual environment
CMD ["./.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]