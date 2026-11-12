FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install ".[serve]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# TABPFN_TOKEN (and optionally LLM_* vars) are provided at runtime:
#   docker run -e TABPFN_TOKEN=... -p 8000:8000 data-science-agent
CMD ["tabular-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
