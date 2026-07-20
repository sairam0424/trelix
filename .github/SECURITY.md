# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.7.x   | ✅ Active |
| 2.6.x   | ⚠️ Security fixes only |
| < 2.6   | ❌ End of life |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

To report a security vulnerability privately:

1. **GitHub Security Advisories (preferred):** Go to [Security → Advisories → New advisory](https://github.com/sairam0424/trelix/security/advisories/new) in this repository
2. **Email:** trelix-security@[maintainer-domain] *(replace with actual contact)*

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
| Public disclosure | After fix is available |

## Security Considerations for Self-Hosted Deployments

- **`trelix serve`** binds to `127.0.0.1` by default — do NOT expose to public internet without authentication
- **`GITHUB_TOKEN`** for `trelix review --pr` should use fine-grained PATs with minimum scope (`pull_requests:write` + `contents:read`)
- **Index files** (`.trelix/index.db`) contain your source code — treat with same sensitivity as source
- **Query telemetry** (`TRELIX_TELEMETRY_ENABLED=true`) stores query text locally — do not enable on sensitive codebases without reviewing the storage implications

## Dependency Security

trelix uses PyPI Trusted Publishing with automatic Sigstore attestations (PEP 740) for all releases. Each release artifact is cryptographically linked to its source commit on the public Sigstore transparency log.

Verify a release:
```bash
pip install pypi-attestations
python -m pypi_attestations verify trelix==2.8.1
```
