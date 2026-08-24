"""
Meta-test for HOLE 1 of the round-4 operator-env scrub: an installed provider SDK
reads its OWN env names out of ``os.environ``, and no amount of asking config.py
can enumerate them.

WHAT WAS OPEN, quoted verbatim from the round-4 author's own note
----------------------------------------------------------------
  "THE 28-NAME TABLE IS SCOPED TO WHAT config.py DECLARES, NOT TO EVERY NAME A
   PROVIDER SDK READS. AWS_SESSION_TOKEN, AWS_DEFAULT_REGION, OPENAI_BASE_URL ..."

``test_env_isolation_covers_config_aliases.py`` derives its expected selection
from ``config.py``'s pydantic model fields. That is the right derivation for the
question it asks and it is structurally incapable of asking this one: there is no
trelix field for ``AWS_SESSION_TOKEN``, so a config-derived guard reports full
coverage while botocore reads the name straight out of the environment. Round 4
had already measured the same shape once -- "Azure credentials are read by the
OpenAI SDK directly from os.environ, not via pydantic env_prefix" -- so this is
the general case of a hole that was known in one instance.

THE SCOPE OF THIS TEST, AND ITS LIMITS. READ THIS BEFORE TRUSTING IT
--------------------------------------------------------------------
A fully general "no installed SDK reads any credential-shaped name we do not
scrub" check is NOT achievable, so the scope is deliberately narrow and stated
here rather than implied. This test covers exactly:

    a quoted, fully upper-case, underscore-containing string literal
    appearing in the source of one of ``_SCANNED_PACKAGES`` as installed in
    THIS environment, whose name starts with one of ``_PROVIDER_FAMILIES``
    and ends with one of ``_CREDENTIAL_SHAPED_SUFFIXES``.

Five limits follow from that, and none of them is a defect to be fixed by
loosening the definition:

1. NAMES ARE FOUND, NOT READS. The scan matches a literal; it does not prove the
   SDK ever reads it, nor that any reachable trelix code path gets there. It is
   therefore a SUPERSET of "read", which is the safe direction for a
   deny-by-default table: over-inclusion costs a harmless ``delenv``,
   under-inclusion is the hole.
2. COMPUTED NAMES ARE INVISIBLE. botocore builds per-service names with
   f-strings -- ``f'AWS_ENDPOINT_URL_{transformed_service_id_env}'``
   (botocore/configprovider.py:1041) and ``f"AWS_BEARER_TOKEN_{bearer_name}"``
   (botocore/utils.py:3638). No literal scan can enumerate those; only the
   ``AWS_ENDPOINT_URL`` stem is caught. Closing that would need a prefix rule for
   ``AWS_ENDPOINT_URL_``/``AWS_BEARER_TOKEN_``, which is a real follow-up and is
   NOT claimed here.
3. INSTALLED PACKAGES ONLY. An optional extra that is not installed in this venv
   contributes nothing, so this test's coverage is a property of the environment
   as much as of the code. ``test_scan_found_every_expected_package`` fails loudly
   rather than silently narrowing if one disappears.
4. THE FAMILY FILTER IS A CHOICE, not a derivation. ``_PROVIDER_FAMILIES`` is
   trelix's own provider surface. A future provider outside those families is
   NOT covered until someone adds the family -- which is why
   ``test_families_cover_the_config_table`` ties the family list back to
   ``CONFIG_NON_PREFIXED_ENV`` so adding a provider to config.py without widening
   the families here fails instead of passing quietly.
5. SHAPE IS BY NAME, NOT BY MEANING. ``AWS_USE_FIPS_ENDPOINT`` ends in
   ``_ENDPOINT`` and is a boolean. It is in the table. That is accepted
   over-inclusion, not a misclassification -- see the rationale block in
   ``tests/_env_isolation.py``.

WHY READING THE SDKs IS NOT A RULE-1 VIOLATION
----------------------------------------------
Same distinction the config meta-test draws. The module under test is
``tests/_env_isolation.py``. The installed SDKs are the SELECTOR -- they decide
WHICH names must be scrubbed. No assertion below reads an expected value out of
them, and the pinned literal that keeps the assertions honest lives in
``_env_isolation.py`` and is compared both ways.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re

import pytest

from tests._env_isolation import (
    CONFIG_NON_PREFIXED_ENV,
    INSTALLED_SDK_PROVIDER_ENV,
    scrub_operator_env,
)

# ---------------------------------------------------------------------------
# The scan definition. This IS the scope; see the docstring.
# ---------------------------------------------------------------------------

# Every provider SDK trelix can talk to that is installed here. Listed
# explicitly rather than walked over all of site-packages: scanning every
# installed distribution would pull in names from packages trelix has no provider
# relationship with, and the resulting table could not be reasoned about.
_SCANNED_PACKAGES: tuple[str, ...] = (
    "anthropic",
    "boto3",
    "botocore",
    "cohere",
    "litellm",
    "openai",
    "qdrant_client",
    "voyageai",
)

# Quoted, all-caps, contains at least one underscore. The underscore requirement
# drops single-word constants ("GET", "POST", "UTF") without needing a stoplist.
_ENV_NAME_LITERAL = re.compile(r"""["']([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)["']""")

# trelix's provider surface. Tied back to CONFIG_NON_PREFIXED_ENV by
# test_families_cover_the_config_table, so this cannot silently fall behind
# config.py. VERTEX_/VERTEXAI_/GEMINI_/CO_ are the SDK-side spellings of families
# config.py names differently (GOOGLE_, COHERE_), which is precisely the class of
# name a config-derived guard cannot see.
_PROVIDER_FAMILIES: tuple[str, ...] = (
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "CO_",
    "COHERE_",
    "GEMINI_",
    "GOOGLE_",
    "LANCE_",
    "OPENAI_",
    "OTEL_",
    "QDRANT_",
    "VERTEX_",
    "VERTEXAI_",
    "VOYAGE_",
)

# Suffix vocabulary = the operational definition of "credential- or
# endpoint-shaped". Anchored at the END on purpose: an unanchored "contains
# _TOKEN" match also selects AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS (a
# price) and "contains _VERSION" selects ANTHROPIC_TOKEN_COUNTING_BETA_VERSION (a
# beta date). Both were in an earlier iteration of this scan and neither is an
# operator credential channel; suffix-anchoring removes them without a stoplist.
_CREDENTIAL_SHAPED_SUFFIXES: tuple[str, ...] = (
    "_ACCESS_KEY_ID",
    "_ACCOUNT_ID",
    "_API_BASE",
    "_API_VERSION",
    "_BASE_URL",
    "_CERTIFICATE_PATH",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_CREDENTIALS_FILE",
    "_ENDPOINT",
    "_KEY",
    "_KEY_ID",
    "_KEY_PATH",
    "_PASSWORD",
    "_PROFILE",
    "_PROFILE_NAME",
    "_REGION",
    "_REGION_NAME",
    "_ROLE_ARN",
    "_SECRET",
    "_SECRET_ACCESS_KEY",
    "_SESSION_NAME",
    "_TOKEN",
    "_TOKEN_FILE",
    "_URI",
    "_URL",
)

# Anti-vacuous floor on the raw literal count, before any filtering. If a future
# refactor breaks the regex or the package discovery, the filtered set goes empty
# and "every scanned name is covered" becomes trivially true -- which is exactly
# the green-when-vacuous failure this whole file exists to close, reappearing one
# level up. Loose on purpose: the point is "the scan is finding thousands of
# literals", not a pinned total that a routine SDK bump would break. 1,673
# measured here, so 1,200 leaves ~28% of slack for an SDK that sheds constants.
_MIN_RAW_LITERALS = 1200

# Same floor for the filtered set: 110 measured here, floor at 100.
_MIN_SELECTED_NAMES = 100


def _package_root(pkg: str) -> pathlib.Path | None:
    """Directory of an installed package, or None if it is not installed."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def _scan() -> tuple[set[str], dict[str, str], set[str]]:
    """Return (all raw literals, selected name -> "pkg/relpath:line", found pkgs).

    The provenance map is what makes a failure actionable: the message names the
    FILE:LINE the offending name was seen at, so the next person does not have to
    re-grep site-packages to decide whether it is real. Names only -- this reads
    package source, never the environment, so no operator value can reach a
    failure message from here.
    """
    raw: set[str] = set()
    selected: dict[str, str] = {}
    found: set[str] = set()

    for pkg in _SCANNED_PACKAGES:
        root = _package_root(pkg)
        if root is None:
            continue
        found.add(pkg)
        for path in sorted(root.rglob("*.py")):
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(src.splitlines(), start=1):
                for match in _ENV_NAME_LITERAL.finditer(line):
                    name = match.group(1)
                    raw.add(name)
                    if name.startswith(_PROVIDER_FAMILIES) and name.endswith(
                        _CREDENTIAL_SHAPED_SUFFIXES
                    ):
                        selected.setdefault(name, f"{pkg}/{path.relative_to(root)}:{lineno}")
    return raw, selected, found


# Scanning ~4,900 files takes ~2.5s. Done once at module scope so the six tests
# below share it rather than paying it each.
_RAW_LITERALS, _SELECTED, _FOUND_PACKAGES = _scan()


def _is_installed(pkg: str) -> bool:
    """True if importlib can resolve ``pkg`` at all, independently of the scan.

    This is the DISCRIMINATOR that separates the two reasons ``_package_root``
    returns None: the package is genuinely not installed (a legitimately thinner
    environment) versus the package is installed but the discovery walk broke (a
    real defect this file must catch). ``_package_root`` collapses both into None,
    so asking importlib a second, narrower question is what keeps the precondition
    from having to guess.
    """
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


# Absent because the environment is thin, versus missing because discovery broke.
_INSTALLED_PACKAGES = frozenset(p for p in _SCANNED_PACKAGES if _is_installed(p))
_ABSENT_PACKAGES = frozenset(_SCANNED_PACKAGES) - _INSTALLED_PACKAGES
_FULL_SET_PRESENT = not _ABSENT_PACKAGES

# The packages MEASURED to declare at least one in-scope env name, so a zero from any
# of them means the scan broke for that package rather than that it has nothing to say.
# boto3 and qdrant_client are deliberately excluded: both measured at exactly 0 and
# both legitimately so (boto3 defers its constants to botocore; qdrant_client reads no
# environment at all). Asserting positivity across ALL located packages is the
# uniform-assertion-over-differing-components mistake, and it failed in the full venv.
_PACKAGES_THAT_MUST_CONTRIBUTE = frozenset(
    {"anthropic", "botocore", "cohere", "litellm", "openai", "voyageai"}
)

# WHY THIS FILE IS ENVIRONMENT-AWARE, and it is a fix rather than a loosening.
#
# The first version of this file asserted all eight packages unconditionally, with
# the reasoning written into its own docstring: "Fails rather than skips: all eight
# are installed in this venv, so a miss means the discovery broke, not that the
# environment is legitimately thinner." True of the venv it was written in, and
# false as a universal. CI's unit job installs
# `.[local,otel,sso,bge-code,nomic-code,dev]`, which brings NONE of anthropic,
# boto3, botocore, cohere, voyageai, qdrant_client or litellm -- so `openai` alone
# was scanned and three tests here failed on all four Python legs while passing
# locally. Same shape as the AZURE_EXTENSION_DIR failure one commit earlier: an
# assertion stronger than the environment guarantees, validated in a rich env.
#
# Reproduced before fixing, not reasoned about: scratch-pad/leanenv/ci_lean_blocker.py
# installs a meta_path finder that DECLINES those seven names, so find_spec raises
# exactly as it does when they are absent. Under it, this file failed the same three
# tests by name.
#
# THE SCRUB ITSELF IS UNAFFECTED by any of this: scrub_operator_env deletes all 124
# names whatever is installed. Only the meta-test's expectations were coupled.
#
# AND THE DIRECTION THAT MATTERS STAYS STRICT EVERYWHERE.
# test_every_scanned_sdk_env_name_is_covered -- "no in-scope SDK name escapes the
# table", the actual hole-closer -- is not environment-coupled at all: a thinner
# scan yields a SUBSET of names, every one of which must still be in the table. It
# is the three PRECONDITIONS that cannot be stated the same way in both
# environments, and only they are conditioned below.


# ---------------------------------------------------------------------------
# Preconditions. Each names the mechanism that could make the coverage test true
# by construction, and fails (or skips) LOUDLY instead of passing vacuously.
# ---------------------------------------------------------------------------


def test_scan_found_every_expected_package() -> None:
    """Precondition: every package that IS installed was located by the scan.

    Names the mechanism that could go vacuous: ``_package_root``. If it silently
    returns None, the scan yields nothing and every coverage assertion below is
    trivially true.

    The assertion is over ``_INSTALLED_PACKAGES``, not over ``_SCANNED_PACKAGES``,
    and that distinction is the whole fix: an installed-but-unfound package means
    discovery BROKE and must fail; an absent one means the environment is thinner,
    which is the ordinary case in CI. Reporting the absentees either way, so a lean
    run says plainly how wide its coverage was rather than looking identical to a
    full one.

    MUTATION that must make this fail: make ``_package_root`` return ``None``
    unconditionally, or point ``_SCANNED_PACKAGES`` at a package that is installed
    under a different import name.
    """
    unfound = sorted(_INSTALLED_PACKAGES - _FOUND_PACKAGES)
    assert unfound == [], (
        f"these provider SDKs ARE installed but were not located by _package_root, "
        f"so discovery is broken rather than the environment being thin: {unfound}. "
        f"The coverage assertions in this file are only as wide as the packages "
        f"actually scanned."
    )
    if _ABSENT_PACKAGES:
        pytest.skip(
            f"not installed here, so outside this run's reach: "
            f"{sorted(_ABSENT_PACKAGES)}. Scanned {sorted(_FOUND_PACKAGES)}. This is "
            f"the expected state in CI's unit job "
            f"(.[local,otel,sso,bge-code,nomic-code,dev] brings none of them) and is "
            f"NOT evidence the table is wrong -- "
            f"test_every_scanned_sdk_env_name_is_covered still holds strictly over "
            f"whatever WAS scanned."
        )


def test_scan_is_not_vacuous() -> None:
    """Precondition: the regex and the file walk are both still finding things.

    Names the mechanism: ``_ENV_NAME_LITERAL`` over ``root.rglob("*.py")``. A
    broken regex or an empty walk makes ``_SELECTED`` empty, and
    "every selected name is covered" passes over nothing -- the round-4a
    green-when-vacuous shape, one level up from where it was found.

    MUTATION that must make this fail: change ``_ENV_NAME_LITERAL`` to match
    ``r"ZZZ_NEVER_MATCHES"``, or change ``rglob("*.py")`` to
    ``rglob("*.nosuchext")``.
    """
    # Always true, whatever is installed: something must have been scanned at all.
    # Without this the two conditional blocks below could both be skipped past on an
    # environment with zero provider SDKs, and "the scan is not vacuous" would itself
    # become vacuous -- the failure this test exists to prevent, one level up again.
    assert _FOUND_PACKAGES, (
        "no provider SDK was located at all, so _SELECTED is empty and every coverage "
        "assertion in this file passes over nothing. Even CI's leanest job installs "
        "openai; zero packages means discovery is broken, not that the env is thin."
    )

    # Per-package, so the check scales with the environment instead of assuming the
    # full set. The absolute floors below were measured over all eight (1,673 raw /
    # 110 selected) and are meaningless when seven are absent: openai alone yields 22.
    #
    # NOT every located package: two contribute zero LEGITIMATELY, measured, and a
    # blanket positivity assertion over all of them failed in the full venv on the
    # first attempt. boto3 is a thin wrapper whose constants live in botocore (0 vs
    # 21), and qdrant_client reads no environment at all -- its QDRANT_* names are
    # trelix-declared aliases only. Requiring positivity from those two would fail
    # forever, so the table below is the measured set that MUST contribute, and a zero
    # from any of them means the walk or the regex broke for that package.
    #   anthropic 18   botocore 21   cohere 3   litellm 61   openai 6   voyageai 1
    barren = sorted(
        pkg
        for pkg in _FOUND_PACKAGES & _PACKAGES_THAT_MUST_CONTRIBUTE
        if not any(loc.startswith(f"{pkg}/") for loc in _SELECTED.values())
    )
    assert barren == [], (
        f"these packages were located and are known to declare env names, but "
        f"contributed none, so the regex or the file walk is broken for them: {barren}"
    )

    if _FULL_SET_PRESENT:
        assert len(_RAW_LITERALS) >= _MIN_RAW_LITERALS, (
            f"only {len(_RAW_LITERALS)} upper-case literals found across "
            f"{sorted(_FOUND_PACKAGES)}; the scan has gone vacuous"
        )
        assert len(_SELECTED) >= _MIN_SELECTED_NAMES, (
            f"only {len(_SELECTED)} provider-family credential-shaped names selected; "
            f"the family or suffix filter has gone vacuous"
        )
    else:
        pytest.skip(
            f"the absolute floors ({_MIN_RAW_LITERALS} raw / {_MIN_SELECTED_NAMES} "
            f"selected) were measured over all {len(_SCANNED_PACKAGES)} SDKs and do not "
            f"apply with {sorted(_ABSENT_PACKAGES)} absent. The per-package check above "
            f"ran and passed over {sorted(_FOUND_PACKAGES)}."
        )


def test_families_cover_the_config_table() -> None:
    """Precondition: ``_PROVIDER_FAMILIES`` has not fallen behind config.py.

    Limit 4 in the module docstring is that the family list is a CHOICE. This is
    what stops that choice from silently going stale: every name config.py
    already declares as non-prefixed must fall under some family here, so adding
    a provider to config.py whose family is not listed fails HERE -- rather than
    quietly leaving that provider's SDK names outside the scan's reach.

    MUTATION that must make this fail: remove ``"VOYAGE_"`` from
    ``_PROVIDER_FAMILIES``.
    """
    unfamilied = sorted(
        name for name in CONFIG_NON_PREFIXED_ENV if not name.startswith(_PROVIDER_FAMILIES)
    )
    assert unfamilied == [], (
        f"config.py declares these non-prefixed env names but no family in "
        f"_PROVIDER_FAMILIES matches them, so this scan cannot see their SDK-side "
        f"siblings: {unfamilied}"
    )


# ---------------------------------------------------------------------------
# The guard: set-equality both ways between the scan and the pinned table.
# ---------------------------------------------------------------------------


def test_every_scanned_sdk_env_name_is_covered() -> None:
    """THE hole-closer: no in-scope SDK name escapes the table.

    This is the direction that matters. An SDK upgrade that introduces a new
    provider credential or endpoint name fails HERE, loudly, instead of becoming
    a silent channel for the operator's environment -- which is what
    ``test_every_config_env_name_is_covered`` structurally could not do, because
    the name has no config.py field to be derived from.

    MUTATION that must make this fail: delete ``"AWS_SESSION_TOKEN"`` (or any
    other entry) from ``INSTALLED_SDK_PROVIDER_ENV`` in
    ``tests/_env_isolation.py``.
    """
    table = {name.upper() for name in INSTALLED_SDK_PROVIDER_ENV}
    uncovered = sorted(name for name in _SELECTED if name not in table)
    detail = "\n".join(f"    {n}  ({_SELECTED[n]})" for n in uncovered)
    assert uncovered == [], (
        f"{len(uncovered)} provider-family credential/endpoint-shaped env names "
        f"appear in installed SDK source but are not scrubbed by the unit suite:\n"
        f"{detail}\n"
        f"Add each to INSTALLED_SDK_PROVIDER_ENV in tests/_env_isolation.py."
    )


def test_sdk_table_has_no_unscanned_entries() -> None:
    """The other direction: the table cannot accumulate names nothing reads.

    Without this, the table only ever grows and a scrub of a name no installed
    SDK mentions any more reads as coverage. An SDK that DROPS a name fails here,
    which is the intended alarm -- someone should delete the stale entry rather
    than leave a table nobody can tell is current.

    MUTATION that must make this fail: add ``"STRIPE_SECRET_KEY"`` to
    ``INSTALLED_SDK_PROVIDER_ENV`` -- note it satisfies neither the family nor
    the scan, so it can only be a stale hand-edit.

    REQUIRES THE FULL PACKAGE SET, and skips loudly without it. This is the one
    direction that genuinely cannot be evaluated in a thin environment: the table
    was derived from a scan of all eight SDKs, so with seven absent every name
    contributed by those seven reads as "stale" when it is simply out of reach. The
    forward direction needs no such condition -- a subset scan still has to be fully
    covered -- which is why only this test carries the gate.

    LIMIT, STATED SO NOBODY MISTAKES THIS FOR CI COVERAGE: no CI job installs all
    eight, so this alarm fires only in a full development venv. That is where an SDK
    bump happens, so it is the right place for it, but CI will never raise it and a
    dropped SDK name can therefore sit unnoticed until someone runs the full set
    locally.
    """
    if not _FULL_SET_PRESENT:
        pytest.skip(
            f"stale-entry detection needs all {len(_SCANNED_PACKAGES)} SDKs; absent "
            f"here: {sorted(_ABSENT_PACKAGES)}. Every name those packages contribute "
            f"would read as stale. Run in a venv with all provider extras to exercise "
            f"this direction."
        )
    stale = sorted(set(name.upper() for name in INSTALLED_SDK_PROVIDER_ENV) - set(_SELECTED))
    assert stale == [], (
        f"INSTALLED_SDK_PROVIDER_ENV lists names the scan of "
        f"{sorted(_FOUND_PACKAGES)} does not find: {stale} -- either an SDK "
        f"dropped them, or they were added by hand without evidence."
    )


def test_sdk_table_is_upper_case_and_duplicate_free() -> None:
    """Storage invariants the scrub depends on.

    ``scrub_operator_env`` uppercases the ENVIRONMENT key and compares against
    the table, so a lowercase entry would never match and would be a silent
    non-scrub. A duplicate is harmless at runtime but hides a botched merge.

    MUTATION that must make this fail: lowercase one entry of
    ``INSTALLED_SDK_PROVIDER_ENV``, or duplicate one.
    """
    assert [n for n in INSTALLED_SDK_PROVIDER_ENV if n != n.upper()] == []
    assert len(INSTALLED_SDK_PROVIDER_ENV) == len(set(INSTALLED_SDK_PROVIDER_ENV))


# ---------------------------------------------------------------------------
# Behavioural guard: the three names the round-4 author named, end to end.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        # The three quoted verbatim in the round-4 note, with the installed-source
        # evidence recorded next to each. Written as an explicit table, NOT
        # iterated out of INSTALLED_SDK_PROVIDER_ENV: iterating the collection
        # under test would make this pass for whatever the table happens to
        # contain, which is the opposite of pinning it.
        ("AWS_SESSION_TOKEN", "credential"),  # botocore/credentials.py:1192
        ("AWS_DEFAULT_REGION", "region"),  # botocore/configprovider.py:74
        ("OPENAI_BASE_URL", "endpoint"),  # openai/_client.py:251
        # One per remaining family, so a family-wide regression is caught too.
        ("AZURE_OPENAI_API_KEY", "credential"),  # openai/__init__.py:342
        ("ANTHROPIC_BASE_URL", "endpoint"),  # anthropic/_client.py:225
        ("CO_API_KEY", "credential"),  # cohere/base_client.py:119
        ("VOYAGE_API_KEY_PATH", "credential-path"),  # voyageai/util.py:78
        ("GEMINI_API_KEY", "credential"),  # litellm/setup_wizard.py:72
    ],
)
def test_scrub_removes_each_named_sdk_channel(
    name: str, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``scrub_operator_env`` actually removes each name, not merely lists it.

    Asserts the THING (the name is gone from ``os.environ`` after the scrub)
    rather than the proxy (the name appears in a tuple). A table entry that the
    scrub never consults -- the exact bug a typo or a missed ``targets |=`` would
    produce -- passes every set-equality test above and fails here.

    Each case is plant-then-scrub, in ITS OWN parametrised run: one name per
    mutation, so a kill is attributable to that name and not to a sibling
    (round-4b's coupled-measurement lesson). The precondition below fails if the
    plant did not take, so a broken ``setenv`` cannot make this pass on nothing.

    Values are inert placeholders and no assertion message renders one:
    ``assert name not in os.environ`` would make pytest dump the whole
    environment, credential values included, into the CI log.

    MUTATION that must make this fail: drop the
    ``targets |= {... INSTALLED_SDK_PROVIDER_ENV}`` line from
    ``scrub_operator_env``, or remove this ``name`` from the table.
    """
    monkeypatch.setenv(name, "planted-inert-placeholder")

    # Precondition, naming the fixture: monkeypatch.setenv. Without it a
    # silently-failed plant leaves nothing to remove and the assertion below is
    # true by construction.
    assert name in os.environ, f"monkeypatch.setenv did not plant {name}; nothing to scrub"

    scrub_operator_env(monkeypatch)

    leaked = sorted(k for k in os.environ if k.upper() == name)
    assert leaked == [], (
        f"{name} ({kind}) survived scrub_operator_env: {leaked}. It is in "
        f"INSTALLED_SDK_PROVIDER_ENV but the scrub is not consulting that table."
    )
