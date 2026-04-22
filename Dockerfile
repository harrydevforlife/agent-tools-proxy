FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Create non-root user early (so we can chown on copy)
RUN useradd -m -u 10001 appuser

# Install uv (fast dependency installer) + minimal runtime deps
RUN pip install --no-cache-dir uv

# Install deps first (better Docker layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy app source
COPY --chown=appuser:appuser app ./app
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Use venv created by uv sync
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

