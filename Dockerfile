# syntax=docker/dockerfile:1
#
# Two published tags share this one Dockerfile via the EXTRAS build arg:
#   docker build --build-arg EXTRAS=serve         -> ghcr.io/sairam0424/trelix:X.Y.Z        (slim, API-embedder-only)
#   docker build --build-arg EXTRAS=serve,local    -> ghcr.io/sairam0424/trelix:X.Y.Z-local  (bundles sentence-transformers/torch)
# See .github/workflows/docker-publish.yml for the release-time build matrix.

FROM python:3.11-slim AS builder

ARG EXTRAS=serve

WORKDIR /build

# build-essential covers any optional-dependency wheel that needs a compiler
# on an architecture without a prebuilt wheel (this stage is never copied
# into the runtime image, so it costs nothing in the final size).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install ".[${EXTRAS}]"

FROM python:3.11-slim AS runtime

RUN groupadd --system trelix \
    && useradd --system --gid trelix --home-dir /home/trelix --create-home trelix

COPY --from=builder /install /usr/local

# Repos are indexed/served from a bind-mounted volume at /repo (see
# docker-compose.yml). The trelix user must own it so `.trelix/` index
# writes succeed — override with `docker run --user "$(id -u):$(id -g)"`
# if your host UID needs to match the mounted volume's ownership instead.
RUN mkdir -p /repo && chown trelix:trelix /repo

WORKDIR /repo
USER trelix

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)" || exit 1

ENTRYPOINT ["trelix"]
CMD ["serve", "/repo", "--host", "0.0.0.0", "--port", "8765"]
