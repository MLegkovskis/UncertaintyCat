FROM python:3.14-slim@sha256:cae66f2ef0ec51a9891263eeee7f987dacf0a9879e8aa9353d5606e0530619a5 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN groupadd --system uncertaintycat \
    && useradd --system --gid uncertaintycat --create-home uncertaintycat

COPY --from=ghcr.io/astral-sh/uv:0.9.9@sha256:f6e3549ed287fee0ddde2460a2a74a2d74366f84b04aaa34c1f19fec40da8652 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock LICENSE ./
COPY uncertaintycat_core ./uncertaintycat_core
COPY services ./services

RUN uv sync --frozen --no-dev --no-install-project \
    && chown -R uncertaintycat:uncertaintycat /app

USER uncertaintycat
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"

CMD ["uvicorn", "services.compute.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
