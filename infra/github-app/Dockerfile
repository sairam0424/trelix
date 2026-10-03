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

FROM node:24-bookworm-slim AS node-deps
WORKDIR /app
COPY infra/github-app/package.json infra/github-app/package-lock.json ./
RUN npm ci --omit=dev

FROM node:24-bookworm-slim AS node-builder
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

# Node 24 (LTS, end of life 2028-04-30; Node 20 reached end of life on
# 2026-04-30) is copied in as one binary from the official node image that the
# two stages above build with, so the service runs on the Node it was built and
# type-checked on. Debian's own `nodejs` apt package is too old (engines pins
# >=24), and the previous NodeSource `curl | bash` setup script ran whatever that
# URL served at build time. The binary is the nodejs.org build, dynamically
# linked against glibc and libstdc++: this stage is Debian trixie (glibc 2.41),
# newer than the bookworm (glibc 2.36) image the binary comes from, so it loads
# here, and the base already ships libstdc++6. npm and npx are not copied: the
# entrypoint is `node dist/server.js` and nothing at runtime calls them.
COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node

# `git` itself — not a Python or Node dependency, but repo-checkout.ts
# shells out to the `git` binary directly (init/remote/fetch/checkout) to
# clone each PR's head. Caught by Phase 5's real
# `docker exec <container> which trelix git` check: the base
# python:3.14-slim/node:24-bookworm-slim images provide neither `trelix`
# (installed above via pip) nor `git` by default. `ca-certificates` is listed
# because git fetches over HTTPS and needs the system CA bundle; python:3.14-slim
# already has it, and naming it keeps that from depending on the base image.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
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
# The service checks out pull requests written by outsiders and runs `trelix index`
# over them. trelix follows symlinks out of the repo by default; confine the walker
# to each checkout so a committed symlink cannot pull files from this host into the
# index. child-env.ts forces the same value for both trelix children.
ENV TRELIX_WALKER_FOLLOW_SYMLINKS=false
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '3000') + '/health', timeout=3)" || exit 1

CMD ["node", "dist/server.js"]
