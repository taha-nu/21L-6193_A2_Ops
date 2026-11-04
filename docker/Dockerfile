# VisionGuard web app (Task 6.2). Build from the repo root:
#   docker build -f docker/Dockerfile -t visionguard-web .
#
# Order of instructions = layer cache strategy. Docker re-uses every layer up to the first
# instruction whose inputs changed. Requirements change rarely, source code changes often,
# so we install dependencies FIRST (one big, cached layer) and copy the code LAST. Editing a
# template then rebuilds in seconds instead of re-downloading TensorFlow.

# ---------- stage 1: build a virtualenv with all dependencies ----------
FROM python:3.12-slim AS builder
ARG TF_PACKAGE=tensorflow-cpu==2.21.0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt "${TF_PACKAGE}"

# ---------- stage 2: slim runtime, no pip cache, no build leftovers ----------
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/champion.keras \
    PORT=8000
# unprivileged user: if the app is compromised the attacker is not root in the container
RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app visionguard/ visionguard/
COPY --chown=app:app app/ app/
COPY --chown=app:app run.py .
# models/ is NOT copied: it is mounted read-only by docker-compose (see docker-compose.yml)
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os,sys; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1
# gunicorn = production WSGI server. 1 worker x 4 threads: each worker holds its own copy
# of the model in RAM, so we scale with threads (TF releases the GIL during inference).
CMD ["sh", "-c", "exec gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 4 --timeout 120 --access-logfile - 'app:create_app()'"]
