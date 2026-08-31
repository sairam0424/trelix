# Security Policy

This is the file GitHub renders in the repository's Security tab — it resolves
`.github/SECURITY.md` ahead of the one in the repository root — so everything a reporter
needs is stated here, with no hop. The **threat model** (what reaches an LLM, prompt
injection, symlink and path-confinement behaviour, per-version security notes) lives in
[`SECURITY.md`](../SECURITY.md) at the repository root and is not repeated here.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.2.4 (latest) | ✅ Fixes ship here |
| everything earlier | ❌ No backports |

Only the most recent release is supported, and that is a description of what this project
has actually done rather than an aspiration. Across all 34 releases from 0.1.0
(2026-06-25) to 3.1.2 (2026-08-17), no patch release has ever landed on an older line
after a newer minor shipped: `2.7.3` came before `2.8.0`, `2.11.1` before `2.12.0`,
`3.0.1` before `3.1.0`. There are no maintenance branches, so there is nothing to
backport to. A security fix is released as a new version from `main`; upgrade to receive
it. The authoritative list of what exists is
[PyPI](https://pypi.org/project/trelix/#history), not this table.

Through v3.1.1 this table claimed `2.7.x` active and `2.6.x` on security fixes only,
while the root `SECURITY.md` simultaneously claimed `1.x` and `0.7.x` — two tables, both
stale, neither listing any 3.x release, and both promising a backport practice that has
never existed.

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

**Email: uggesairam0000@gmail.com** — this is the only channel that works today.

GitHub's private vulnerability reporting is **not enabled on this repository**, so the
Security tab shows no "Report a vulnerability" button and `/security/advisories/new`
will not take a submission from anyone without write access. Measured at 3.1.2:

```bash
gh api repos/sairam0424/trelix/private-vulnerability-reporting   # => {"enabled":false}
```

A maintainer can turn it on with
`gh api --method PUT repos/sairam0424/trelix/private-vulnerability-reporting`; until then,
email is the whole disclosure path. Through v3.1.1 this section advertised advisories as
the *preferred* route and gave `trelix-security@[maintainer-domain]` — a literal
placeholder, shipped with its own "replace with actual contact" note — as the fallback.
Both were dead, so the advertised disclosure path reached nobody.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Affected versions
- Any suggested mitigations

### Response timeline

| Action | Target |
|--------|--------|
| Acknowledgement | 2 business days |
| Initial assessment | 5 business days |
| Fix / workaround | Depends on severity |
| Public disclosure | After a fix is available, on a coordinated timeline (typically 90 days) |

Reporters are credited in the release notes unless they ask not to be.

## Security Considerations for Self-Hosted Deployments

- **`trelix serve`** binds to `127.0.0.1` by default, and the API is **open when no
  credential is configured** — `TRELIX_API_AUTH_TOKEN` unset with SSO off means every
  route answers without authentication. That is safe on loopback and not otherwise. Set
  `TRELIX_API_AUTH_TOKEN=<secret>` (sent as `X-Trelix-Api-Key`) or configure OIDC before
  binding anywhere reachable; `serve` now prints a warning if you do the latter without
  the former.
- **The published container overrides that default.** `Dockerfile` runs
  `serve /repo --host 0.0.0.0`, which is required for a published port to reach the
  process. `docker-compose.yml` therefore publishes to `127.0.0.1:8765` rather than all
  interfaces — widen it only together with `TRELIX_API_AUTH_TOKEN`.
- **`GITHUB_TOKEN`** for `trelix review --pr` should use fine-grained PATs with minimum scope (`pull_requests:write` + `contents:read`)
- **Index files** (`.trelix/index.db`) contain your source code — treat with same sensitivity as source
- **Query telemetry** (`TRELIX_TELEMETRY_ENABLED=true`) stores query text locally — do not enable on sensitive codebases without reviewing the storage implications

## Dependency Security

trelix publishes to PyPI via Trusted Publishing, and every distribution carries a
Sigstore-backed PEP 740 attestation binding it to the source commit and tag it was built
from. The `publish` job in `.github/workflows/release.yml` is the only job granted
`id-token: write`, and passes `attestations: true` for all four packages (`trelix`,
`trelix-mcp`, `trelix-langchain`, `trelix-llama-index`).

Verify a release:

```bash
pip install pypi-attestations
pypi-attestations verify pypi \
  --repository https://github.com/sairam0424/trelix \
  pypi:trelix-3.1.2-py3-none-any.whl
```

`verify pypi` needs `--repository` and a concrete distribution — a `pypi:`-prefixed
filename, a local path, or a `files.pythonhosted.org` URL. Through v3.1.1 this section
documented `python -m pypi_attestations verify trelix==2.12.0`, which is not a valid
invocation at any version: `verify` takes an `attestation`/`pypi` subcommand, and no form
of it accepts a `name==version` requirement string.

The attestations are real, not just configured. For the last published release:

```bash
curl -s https://pypi.org/integrity/trelix/3.1.1/trelix-3.1.1-py3-none-any.whl/provenance
```

returns a populated `attestation_bundles` naming publisher `sairam0424/trelix`, workflow
`release.yml`, a certificate identity bound to `refs/tags/v3.1.1`, and a
`rekor.sigstore.dev` transparency-log entry with an inclusion proof.
