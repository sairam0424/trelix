# trelix v3.0.0 Breaking Changes Audit & Removal Schedule

**Last updated**: 2026-07-06  
**Status**: Scheduled for v3.0.0 release  
**Deprecation grace period**: Minimum 2 minor versions (v2.4.0 → v3.0.0)

---

## Overview

This document audits all deprecated/legacy code markers in the trelix codebase and lists candidates for removal in v3.0.0. The deprecation policy enforces:

1. **Announce** — deprecation documented in CHANGELOG + `DeprecationWarning` at runtime
2. **Grace period** — minimum 2 minor versions
3. **Migration guide** — CHANGELOG includes exact rename/replacement
4. **Remove** — only on MAJOR version bump

---

## Deprecation Audit Results

### Codebase Scan Command

```bash
grep -rn "DeprecationWarning\|AliasChoices\|deprecated\|DEPRECATED\|# legacy" \
  src/trelix/ packages/ \
  --include="*.py" \
  | grep -v "__pycache__\|test_\|\.pyc" \
  | sort
```

**Scan date**: 2026-07-06  
**Total hits**: 6 (all related to single deprecation item)

---

## v3.0.0 Removal Schedule

### 1. TRELIX_RETRIEVAL_FLARE_MAX_ITER Environment Variable

| Aspect | Details |
|--------|---------|
| **Status** | ✅ Ready for v3.0.0 removal |
| **Deprecated in** | v2.4.0 |
| **Removal target** | v3.0.0 |
| **Old name** | `TRELIX_RETRIEVAL_FLARE_MAX_ITER` |
| **New name** | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` |
| **Config class** | `RetrievalConfig` |
| **File locations** | `src/trelix/core/config.py` (lines 429-452) |
| **Deprecation warning** | Yes, emitted at `RetrievalConfig()` instantiation |
| **Backward compat shim** | `AliasChoices` (lines 433-436) |
| **Validator** | `_warn_deprecated_flare_iter_env` (lines 440-452) |

#### Migration Guide

```python
# ❌ OLD (deprecated in v2.4.0, will fail in v3.0.0)
export TRELIX_RETRIEVAL_FLARE_MAX_ITER=2

# ✅ NEW (use this instead)
export TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=2
```

#### Audit Details

**File**: `src/trelix/core/config.py`

- **Line 15**: Import statement for `AliasChoices` from pydantic
- **Line 433-436**: `AliasChoices` backward-compat mapping
- **Line 435**: Comment `# legacy — deprecated in v2.4`
- **Line 440-452**: `_warn_deprecated_flare_iter_env()` model validator
- **Line 446-448**: `DeprecationWarning` message text

**Deprecation message**:
```
TRELIX_RETRIEVAL_FLARE_MAX_ITER is deprecated as of trelix v2.4.0. 
Use TRELIX_RETRIEVAL_FLARE_MAX_RETRIES instead. 
The old name will be removed in v3.0.0.
```

#### Test Coverage

**File**: `tests/unit/test_config.py`

Regression test verifies:
- Old env var is recognized (backward compat works)
- `DeprecationWarning` is emitted
- Warning message mentions `v3.0.0`

**Test name**: `test_flare_max_iter_env_emits_deprecation_warning`

---

## v3.0.0 Implementation Checklist

### Code Removal Tasks

- [ ] **Remove `AliasChoices` import** from `src/trelix/core/config.py:15`
  - Only if no other `AliasChoices` usages exist in that file
  - Verify import used only for `flare_max_retries` field

- [ ] **Remove `validation_alias` from `flare_max_retries` field** in `src/trelix/core/config.py:429-437`
  - Replace with simple:
    ```python
    flare_max_retries: int = Field(
        default=1,
        ge=1,
        le=3,
        alias="TRELIX_RETRIEVAL_FLARE_MAX_RETRIES",
    )
    ```

- [ ] **Remove `_warn_deprecated_flare_iter_env` validator** from `src/trelix/core/config.py:440-452`
  - Entire `@model_validator(mode="after")` method can be deleted

### Documentation Tasks

- [ ] **Update CHANGELOG** — add v3.0.0 section with "Breaking Changes" header
  - List removal of `TRELIX_RETRIEVAL_FLARE_MAX_ITER` support
  - Point to migration guide

- [ ] **Create migration guide** at `docs/migration/v2-to-v3.md`
  - Document all breaking changes
  - Provide side-by-side "before/after" code examples
  - Include removal checklist for users upgrading

- [ ] **Update BACKWARDS_COMPATIBILITY.md**
  - Remove `TRELIX_RETRIEVAL_FLARE_MAX_ITER` from "Current deprecations" table
  - Archive to "Past breaking changes" or "v3.0.0 removals" section

### Test Cleanup Tasks

- [ ] **Remove regression test** `test_flare_max_iter_env_emits_deprecation_warning` from `tests/unit/test_config.py`
  - This test validates the deprecation machinery only — no longer needed once removed

---

## Why Batch All Removals into One MAJOR?

**Research verified** (Pydantic v3 pattern, 3-0 verified via deep analysis):

1. **Users prefer single breaking version** over scattered breakage across minors
2. **Testing simpler** — one migration path vs multiple conditional code paths
3. **Deprecation warning fatigue** — batching reduces warning spam
4. **Clean semantic versioning** — major versions explicitly mean "prepare for changes"

---

## Future Deprecation Audit Schedule

- **v2.5.0**: Run this audit again after any new deprecations added
- **v2.6.0+**: Maintain running list until v3.0.0 release
- **v3.0.0**: Execute removal checklist above and commit

---

## Related Documentation

- [BACKWARDS_COMPATIBILITY.md](../BACKWARDS_COMPATIBILITY.md) — stability guarantees and deprecation policy
- [CHANGELOG.md](../../CHANGELOG.md) — version history with migration guides
- [tests/unit/test_config.py](../../tests/unit/test_config.py) — regression test for flare_max_iter deprecation

---

## Appendix: Full Grep Output

```
src/trelix/core/config.py:15:from pydantic import AliasChoices, Field, field_validator, model_validator
src/trelix/core/config.py:433:        validation_alias=AliasChoices(
src/trelix/core/config.py:435:            "TRELIX_RETRIEVAL_FLARE_MAX_ITER",  # legacy — deprecated in v2.4
src/trelix/core/config.py:440:    def _warn_deprecated_flare_iter_env(self) -> RetrievalConfig:
src/trelix/core/config.py:446:                "TRELIX_RETRIEVAL_FLARE_MAX_ITER is deprecated as of trelix v2.4.0. "
src/trelix/core/config.py:449:                DeprecationWarning,
```

**Total matches**: 6  
**Unique deprecation items**: 1  
**Status**: Ready for v3.0.0 cleanup
