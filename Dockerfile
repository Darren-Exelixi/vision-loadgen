# vision-loadgen image. One build, two uses:
#   - standalone runner for runs that target several workers at once (docker/docker-compose.yml);
#   - /dist/vision_loadgen, the bare package that worker Dockerfiles copy next to vision_shared.
# To release a new version:
#   docker build -t ghcr.io/exelixi-ai/vision-loadgen:0.1.0 .
#   docker push ghcr.io/exelixi-ai/vision-loadgen:0.1.0
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /vision-loadgen-src/
COPY vision_loadgen /vision-loadgen-src/vision_loadgen
RUN pip install --no-cache-dir "/vision-loadgen-src[standalone]" \
    && pip install --no-cache-dir --no-deps --no-compile --target /dist /vision-loadgen-src \
    && rm -rf /dist/*.dist-info /dist/bin

WORKDIR /loadgen
ENTRYPOINT ["python", "-m", "vision_loadgen"]
CMD ["--help"]
