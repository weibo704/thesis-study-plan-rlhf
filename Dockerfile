# Serving image for the student study.
#
# Only the frozen artefacts the request path reads are copied in (~8 MB).
# data/catalog/llm_cache/ is 248 MB of build-time replay cache and is
# deliberately excluded: a student's query is new, so it would miss anyway.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first, so a code change does not re-resolve the wheel set.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Code.
COPY pipeline/ ./pipeline/
COPY app/ ./app/
COPY config/ ./config/

# Frozen data the serving path reads. Listed file by file rather than
# copying data/ wholesale, so a future 200 MB artefact cannot silently
# end up in the image.
COPY data/catalog/courses.parquet            ./data/catalog/
COPY data/catalog/skills.parquet             ./data/catalog/
COPY data/catalog/course_skills_final.parquet ./data/catalog/
COPY data/agent/arm_features.parquet         ./data/agent/
COPY data/agent/embeddings.parquet           ./data/agent/
COPY data/agent/arm_index.parquet            ./data/agent/
COPY data/agent/svd_projection_svd64-v1.npy  ./data/agent/
COPY data/agent/arm_feature_spec.yaml        ./data/agent/
COPY data/agent/reward_model_poc-v6-stratified.json ./data/agent/
COPY data/world/study_plans.parquet          ./data/world/
COPY data/world/study_plan_items.parquet     ./data/world/
COPY data/world/preregistrations/            ./data/world/preregistrations/
COPY reports/manifest.json                   ./reports/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv
USER appuser

ENV THESIS_PUBLIC=1 \
    THESIS_STORE=dynamodb \
    PORT=8080
EXPOSE 8080

# Two workers, preloaded: the catalog is built once before the fork and
# shared copy-on-write instead of loaded twice. The timeout is generous
# because a plan generation is a model call, not a database read.
CMD exec gunicorn --bind "0.0.0.0:${PORT}" \
      --workers 2 --threads 4 --preload \
      --timeout 120 --graceful-timeout 30 \
      --access-logfile - --error-logfile - \
      app.wsgi:app
