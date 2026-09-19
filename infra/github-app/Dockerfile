# syntax=docker/dockerfile:1
#
# Build context is the REPO ROOT (not infra/github-app/) — the Python
# stage needs `src/` for the `trelix` CLI this service shells out to.
# Build from repo root:
#   docker build -f infra/github-app/Dockerfile -t trelix-github-app .
#
# Four stages: two build the TypeScript service (deps-only vs. full+build,
# so the runtime image never carries devDependencies), one builds the
# Python `trelix` CLI the service shells out to, and `runtime` combines
# both — this image needs Node (to run the Express service) AND Python +
# `trelix` (to run `trelix index`/`trelix review` as subprocesses).

FROM node:20-bookworm-slim AS node-deps
WORKDIR /app
COPY infra/github-app/package.json infra/github-app/package-lock.json ./
RUN npm ci --omit=dev

FROM node:20-bookworm-slim AS node-builder
WORKDIR /app
COPY infra/github-app/package.json infra/github-app/package-lock.json ./
RUN npm ci
COPY infra/github-app/ .
RUN npm run build

FROM python:3.14-slim AS python-deps
WORKDIR /build

# build-essential covers any optional-dependency wheel that needs a
# compiler on an architecture without a prebuilt wheel (this stage is
# never copied into the runtime image, so it costs nothing in the final
# size) — mirrors the root Dockerfile's own builder stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Bare install — no `[serve]`/`[local]` extras. This service only ever
# shells out to `trelix index`/`trelix review --json` (CLI commands), never
# `trelix serve`, and never the local sentence-transformers embedder — the
# base `dependencies` list already includes the OpenAI-API embedder
# (openai/tiktoken) the CLI needs. Skipping extras keeps this image well
# under Render free tier's 512MB RAM cap by never installing
# sentence-transformers/torch.
RUN pip install --no-cache-dir --prefix=/python-install .

FROM python:3.14-slim AS runtime

# Same CVE mitigation as the root Dockerfile's runtime stage, for the same
# reason: this stage shares that exact base image (python:3.14-slim), so
# it inherits the same unpatched-in-upstream-image vulnerability until the
# base image itself is rebuilt. Re-verify this is still the current
# outstanding issue before reusing this pin on a future rebuild — this
# comment is already dated the moment it's written, same as the root
# Dockerfile's own equivalent comment.
#
# CVE-2026-14456 (libssl3t64 / openssl / openssl-provider-legacy):
# python:3.14-slim's plain trixie/main apt suite still ships the
# vulnerable 3.5.6-1~deb13u2 build. Debian's trixie-security channel has
# already published the fix (3.5.7-1~deb13u2) but the upstream base image
# has not been rebuilt against it yet, so pull it explicitly here instead
# of waiting on an image rebuild or doing a blanket `apt-get upgrade -y`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3t64 \
        openssl \
        openssl-provider-legacy \
    && rm -rf /var/lib/apt/lists/*

# Debian trixie/bookworm's own `nodejs` apt package is too old for this
# service (needs Node 20+, engines pins >=20) — NodeSource's setup script
# installs a current Node 20 build instead.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# `git` itself — not a Python or Node dependency, but repo-checkout.ts
# shells out to the `git` binary directly (init/remote/fetch/checkout) to
# clone each PR's head. Caught by Phase 5's real
# `docker exec <container> which trelix git` check: the base
# python:3.14-slim/node:20-bookworm-slim images provide neither `trelix`
# (installed above via pip) nor `git` by default.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system trelix \
    && useradd --system --gid trelix --home-dir /home/trelix --create-home trelix

COPY --from=python-deps /python-install /usr/local

# GHSA-6v7p-g79w-8964 (msgpack 1.1.2) and CVE-2025-47273 (setuptools
# 70.3.0) are pip's own internally vendored copies, not real trelix
# dependencies — see the root Dockerfile's identical comment/rationale.
# This image's entrypoint is `node dist/server.js`, which never invokes
# pip either, so drop it (and its vendored copies) entirely.
RUN python -m pip uninstall --yes pip setuptools wheel

WORKDIR /app
COPY --from=node-deps /app/node_modules ./node_modules
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/package.json ./package.json

RUN chown -R trelix:trelix /app
USER trelix

ARG PORT=3000
ENV PORT=${PORT}
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '3000') + '/health', timeout=3)" || exit 1

CMD ["node", "dist/server.js"]
