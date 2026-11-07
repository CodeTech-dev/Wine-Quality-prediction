FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

COPY predict.py serve.py model.bin ./

EXPOSE 9696

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]