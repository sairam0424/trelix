# syntax=docker/dockerfile:1
#
# Two published tags share this one Dockerfile via the EXTRAS build arg:
#   docker build --build-arg EXTRAS=serve         -> ghcr.io/sairam0424/trelix:X.Y.Z        (slim, API-embedder-only)
#   docker build --build-arg EXTRAS=serve,local    -> ghcr.io/sairam0424/trelix:X.Y.Z-local  (bundles sentence-transformers/torch)
# See .github/workflows/docker-publish.yml for the release-time build matrix.

FROM python:3.14-slim AS builder

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
# trelix-mcp was never copied into this image, so the published tags shipped no
# trelix-mcp console script at all — found by the 3.2.1 production-verification audit
# (docker run --entrypoint trelix-mcp <image> --help failed with exit 127, confirmed
# genuinely absent from console_scripts, not a PATH issue). Installed in the SAME pip
# invocation as core below so the resolver dedupes the shared trelix requirement in one
# pass rather than risking two separate --prefix installs disagreeing on it.
COPY packages/trelix-mcp/ packages/trelix-mcp/

# --extra-index-url pulls torch's CPU-only wheels first (this image never
# has GPU access) — the default PyPI torch wheel drags in the full CUDA/
# NVIDIA runtime (nvidia-cublas, nvidia-cudnn, cuda-toolkit, ...), which
# blew past hosted-runner disk space during the linux/arm64 (-local variant)
# release build. Harmless no-op for the slim (EXTRAS=serve) build, which
# never installs torch at all.
RUN pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[${EXTRAS}]" ./packages/trelix-mcp

FROM python:3.14-slim AS runtime

# CVE-2026-14456 (libssl3t64 / openssl / openssl-provider-legacy): python:3.14-slim's
# plain trixie/main apt suite still ships the vulnerable 3.5.6-1~deb13u2 build.
# Debian's trixie-security channel has already published the fix
# (3.5.7-1~deb13u2) but the upstream base image has not been rebuilt against it
# yet, so pull it explicitly here instead of waiting on an image rebuild or doing
# a blanket `apt-get upgrade -y`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3t64 \
        openssl \
        openssl-provider-legacy \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system trelix \
    && useradd --system --gid trelix --home-dir /home/trelix --create-home trelix

COPY --from=builder /install /usr/local

# GHSA-6v7p-g79w-8964 (msgpack 1.1.2) and CVE-2025-47273 (setuptools 70.3.0) are
# not real trelix/trelix-mcp dependencies — trelix never depends on either
# package. Both are pip's own internally vendored copies
# (pip/_vendor/msgpack, pip/_vendor/pkg_resources), declared in pip's bundled
# bom.cdx.json SBOM and picked up by Trivy's Python-package scanner. This
# runtime image's ENTRYPOINT is the `trelix` console script and never invokes
# pip, so drop pip (and its vendored msgpack/setuptools copies) from the
# shipped image entirely.
RUN python -m pip uninstall --yes pip setuptools wheel

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
