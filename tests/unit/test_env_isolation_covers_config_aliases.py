"""
Meta-test: the unit suite's env isolation must cover every env name
``trelix.core.config`` reads. Fails when config.py gains an alias the isolation
does not scrub.

WHY READING config.py IS CORRECT HERE, AND NOT A RULE-1 VIOLATION
-----------------------------------------------------------------
The house rule is "never import or recompute the expected value from the module
under test". The module under test here is ``tests/_env_isolation.py``, not
``config.py``. config.py is the *selector*: it decides WHICH names have to be
scrubbed. It never supplies an expected value -- no assertion below reads a
field default, a provider name or a credential out of it. That distinction is
the whole point of a meta-test: an isolation table hand-copied from config.py
drifts the moment someone adds a field, and a test that hard-codes the same
hand-copy drifts with it. Deriving the SELECTION at runtime is what makes drift
impossible; the two ``_EXPECTED_*`` literals below are what keep the ASSERTIONS
honest.

So there are two independent guards, and they fail on different mistakes:

* ``test_non_prefixed_table_matches_literal`` / ``test_scrub_prefixes_match_literal``
  compare the isolation tables against literals written out here. These catch
  someone silently shrinking the isolation.
* ``test_every_config_env_name_is_covered`` compares the isolation against what
  config.py actually declares, at runtime. This catches a NEW alias.

Delete either one and the pair stops working: the literals alone go stale, and
the runtime derivation alone can be satisfied by emptying both sides.

HOW THE ENV NAMES ARE DERIVED
-----------------------------
Not by re-implementing pydantic-settings' naming rules -- that reimplementation
is where the previous audit went wrong (it counted construction sites and missed
that ``env_prefix``-based settings have no ``alias=`` to grep for at all, which
is why it reported ~25 covered names against 106 aliases and still understated
the gap). Instead the authority is asked directly:
``EnvSettingsSource(cls)._extract_field_info(field, name)`` is the exact call
pydantic-settings makes when it resolves a field from the environment, so it
returns the env names that are genuinely live -- ``env_prefix`` + field name for
plain fields, the literal ``alias=`` / ``validation_alias`` for aliased ones,
and every member of an ``AliasChoices``.
"""

from __future__ import annotations

import inspect
import os

import pytest
from pydantic_settings import BaseSettings
from pydantic_settings.sources import EnvSettingsSource

from tests import _env_isolation
from tests._env_isolation import (
    CONFIG_NON_PREFIXED_ENV,
    SCRUB_PREFIXES,
    scrub_operator_env,
)
from trelix.core import config as config_module

# ---------------------------------------------------------------------------
# Literal expectations. Written out by hand, NOT derived -- see module docstring.
# ---------------------------------------------------------------------------

_EXPECTED_PREFIXES = {"TRELIX_"}

# Every env name config.py reads that is NOT under a TRELIX_ prefix, i.e. the
# provider-SDK-conventional names an operator has in their shell or dotenv for
# reasons unrelated to trelix.
_EXPECTED_NON_PREFIXED = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_API_KEY",
    "AZURE_API_VERSION",
    "AZURE_CHAT_MODEL",
    "AZURE_EMBEDDINGS_MODEL",
    "AZURE_ENDPOINT",
    "COHERE_API_KEY",
    "COHERE_ENDPOINT",
    "COHERE_MODEL_RERANK",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_PROJECT",
    "LANCE_TABLE",
    "LANCE_URI",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_PREFER_GRPC",
    "QDRANT_TIMEOUT",
    "QDRANT_URL",
    "VOYAGE_API_KEY",
}

# Floor on the derived name count. Guards the runtime derivation against going
# vacuous: if a pydantic-settings upgrade changes _extract_field_info's shape so
# it yields nothing, "every derived name is covered" becomes trivially true and
# this whole file would pass while covering zero names. Deliberately a loose
# floor, not the exact count -- config.py legitimately gains fields, and pinning
# the total would turn every new field into a failure of THIS test rather than of
# the coverage check that is supposed to catch it.
_MIN_DERIVED_ENV_NAMES = 200


def _settings_classes() -> list[type[BaseSettings]]:
    """Every BaseSettings subclass DEFINED in trelix.core.config."""
    return [
        obj
        for obj in vars(config_module).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseSettings)
        and obj is not BaseSettings
        # __module__ check, not just issubclass: config.py imports names from
        # elsewhere, and a re-exported settings class from another module is not
        # this file's contract to cover.
        and obj.__module__ == config_module.__name__
    ]


def _derived_env_names() -> set[str]:
    """Uppercased env names pydantic-settings will look up for config.py."""
    assert hasattr(EnvSettingsSource, "_extract_field_info"), (
        "pydantic_settings.sources.EnvSettingsSource._extract_field_info is gone; "
        "this meta-test's derivation must be re-pointed at whatever replaced it "
        "rather than deleted"
    )
    names: set[str] = set()
    for cls in _settings_classes():
        source = EnvSettingsSource(cls)
        for field_name, field_info in cls.model_fields.items():
            for _key, env_name, _complex in source._extract_field_info(field_info, field_name):
                names.add(env_name.upper())
    return names


def _is_prefix_covered(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _EXPECTED_PREFIXES)


# ---------------------------------------------------------------------------
# Guard 1 -- the isolation tables against literals.
# ---------------------------------------------------------------------------


def test_scrub_prefixes_match_literal() -> None:
    """SCRUB_PREFIXES is exactly {"TRELIX_"}, compared both ways.

    MUTATION that must make this fail: change ``SCRUB_PREFIXES`` in
    ``tests/_env_isolation.py`` to ``()`` or to
    ``("TRELIX_RETRIEVAL_",)``.
    """
    actual = set(SCRUB_PREFIXES)
    assert actual - _EXPECTED_PREFIXES == set(), "unexpected scrub prefix"
    assert _EXPECTED_PREFIXES - actual == set(), "scrub prefix went missing"
    assert len(SCRUB_PREFIXES) == len(actual), "duplicate entry in SCRUB_PREFIXES"


def test_non_prefixed_table_matches_literal() -> None:
    """CONFIG_NON_PREFIXED_ENV equals the literal above, compared both ways.

    MUTATION that must make this fail: delete ``AZURE_API_KEY`` from
    ``CONFIG_NON_PREFIXED_ENV`` in ``tests/_env_isolation.py``.
    """
    actual = set(CONFIG_NON_PREFIXED_ENV)
    assert actual - _EXPECTED_NON_PREFIXED == set(), "entry not in the pinned literal"
    assert _EXPECTED_NON_PREFIXED - actual == set(), "pinned name missing from the table"
    assert len(CONFIG_NON_PREFIXED_ENV) == len(actual), "duplicate entry"


def test_non_prefixed_table_is_upper_case() -> None:
    """Entries are stored uppercase, because the scrub compares on ``.upper()``.

    A lowercase entry would silently never match, which is the failure mode this
    catches -- ``scrub_operator_env`` uppercases the environment key, not the
    table entry.

    MUTATION that must make this fail: lowercase one entry of
    ``CONFIG_NON_PREFIXED_ENV``.
    """
    assert [n for n in CONFIG_NON_PREFIXED_ENV if n != n.upper()] == []


# ---------------------------------------------------------------------------
# Guard 2 -- the isolation against what config.py declares, at runtime.
# ---------------------------------------------------------------------------


def test_derivation_is_not_vacuous() -> None:
    """Precondition for ``test_every_config_env_name_is_covered``.

    Names the mechanism that could make it true by construction: the
    ``_extract_field_info`` derivation in ``_derived_env_names``. An empty or
    near-empty derived set makes "every derived name is covered" pass without
    covering anything.

    MUTATION that must make this fail: make ``_settings_classes`` return ``[]``.
    """
    derived = _derived_env_names()
    assert len(_settings_classes()) >= 15, "config.py's settings classes were not found"
    assert len(derived) >= _MIN_DERIVED_ENV_NAMES, (
        f"only {len(derived)} env names derived from config.py; the derivation "
        f"has gone vacuous and the coverage check below proves nothing"
    )


def test_every_config_env_name_is_covered() -> None:
    """No env name config.py reads escapes the unit suite's isolation.

    THIS is the anti-drift guard: add a field to any settings class in
    config.py, or a new non-prefixed ``alias=``, and it fails here rather than
    silently becoming a channel for the operator's environment.

    MUTATION that must make this fail: add
    ``newrelic_key: str | None = Field(default=None, alias="NEW_RELIC_KEY")``
    to ``EmbedderConfig``, or drop ``"TRELIX_"`` from ``SCRUB_PREFIXES``.
    """
    derived = _derived_env_names()
    uncovered = sorted(
        name
        for name in derived
        if not _is_prefix_covered(name) and name not in set(CONFIG_NON_PREFIXED_ENV)
    )
    assert uncovered == [], (
        f"config.py reads these env names but the unit suite does not scrub them: "
        f"{uncovered} -- add each to CONFIG_NON_PREFIXED_ENV in "
        f"tests/_env_isolation.py (and to _EXPECTED_NON_PREFIXED here)"
    )


def test_non_prefixed_table_has_no_stale_entries() -> None:
    """The other direction: nothing in the table has left config.py.

    Set-equality with the derived set, so a removed alias is a failure too --
    otherwise the table only ever grows and a scrub of a name nothing reads any
    more looks like coverage.

    MUTATION that must make this fail: add ``"STRIPE_API_KEY"`` to
    ``CONFIG_NON_PREFIXED_ENV``.
    """
    derived_non_prefixed = {name for name in _derived_env_names() if not _is_prefix_covered(name)}
    stale = sorted(set(CONFIG_NON_PREFIXED_ENV) - derived_non_prefixed)
    assert stale == [], f"CONFIG_NON_PREFIXED_ENV scrubs names config.py no longer reads: {stale}"


# ---------------------------------------------------------------------------
# Guard 3 -- the scrub actually works, on a planted leak.
# ---------------------------------------------------------------------------


def test_scrub_removes_a_planted_provider_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """``scrub_operator_env`` neutralizes exactly what ``load_dotenv`` injects.

    Self-contained: it plants the leak itself rather than relying on an ancestor
    ``.env`` existing on the machine, so it discriminates on a clean checkout and
    in CI too. ``"local"`` is a literal, not read back from ``EmbedderConfig``.

    MUTATION that must make this fail: make ``scrub_operator_env`` a no-op
    ``return``.
    """
    monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "azure")
    monkeypatch.setenv("TRELIX_LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://planted.invalid")
    monkeypatch.setenv("AZURE_API_KEY", "planted-not-a-real-key")

    # Precondition: the plant took, so the assertions below have something to
    # remove. Without this the test passes on a machine where setenv silently
    # failed and proves nothing.
    from trelix.core.config import EmbedderConfig

    assert EmbedderConfig().provider == "azure", "the planted leak did not take effect"

    scrub_operator_env(monkeypatch)

    assert EmbedderConfig().provider == "local"

    # Assert what the scrub PROMISES, not a stronger proposition it never made.
    #
    # This line originally read:
    #     assert sorted(k for k in os.environ if k.upper().startswith("AZURE_")) == []
    # and CI falsified it on all four Python legs while it passed on a laptop. GitHub's
    # ubuntu runners preinstall the Azure CLI, which exports
    # AZURE_EXTENSION_DIR=/opt/az/azcliextensions -- a filesystem path, not a credential,
    # declared by no trelix config field, and therefore read by no pydantic source. The
    # assertion had claimed the whole third-party AZURE_ namespace, which the scrub
    # deliberately does NOT: for non-prefixed providers it deletes a NAMED TABLE, precisely
    # because AZURE_/AWS_/OPENAI_ are namespaces trelix does not own. Widening the scrub to
    # match the assertion would have been the wrong repair -- it would delete an unrelated
    # tool's configuration to satisfy a test.
    #
    # Kept instead, in two parts: the planted names must be gone (the scrub works), and no
    # AZURE_ name that config.py DECLARES may remain (the breadth worth keeping -- a real
    # alias the table missed still fails here). Reading the declared names SELECTS what to
    # check and supplies no expected value, so rule 1 holds for the same reason it does in
    # the two guards above.
    #
    # Both comparisons reduce to a list of NAMES before asserting: `assert x not in
    # os.environ` makes pytest render the entire environment, values included, into the CI
    # log.
    planted_left = sorted(k for k in os.environ if k.upper() in {"AZURE_ENDPOINT", "AZURE_API_KEY"})
    assert planted_left == [], f"the scrub left the planted names in place: {planted_left}"

    declared_azure = {n.upper() for n in _derived_env_names() if n.upper().startswith("AZURE_")}
    assert declared_azure, (
        "config.py declares no AZURE_ env name any more, so the check below would be "
        "vacuous; delete it or re-point it at whatever replaced those aliases"
    )
    declared_left = sorted(k for k in os.environ if k.upper() in declared_azure)
    assert declared_left == [], f"the scrub left declared Azure aliases in place: {declared_left}"


def test_scrub_does_not_touch_unrelated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scrub is targeted: a name outside the prefixes and table survives.

    Deny-by-default must not become delete-everything -- ``PATH``, ``HOME`` and
    ``PYTEST_CURRENT_TEST`` have to survive or the suite cannot run at all.

    MUTATION that must make this fail: change ``scrub_operator_env`` to clear
    ``os.environ`` wholesale.
    """
    monkeypatch.setenv("A_HARMLESS_UNRELATED_VAR", "keep-me")
    scrub_operator_env(monkeypatch)
    assert os.environ.get("A_HARMLESS_UNRELATED_VAR") == "keep-me"
    assert os.environ.get("PATH") not in (None, "")


def test_env_file_none_is_not_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the fact that killed the earlier audit's proposed fix.

    ``_env_file=None`` removes pydantic-settings' dotenv-file source, but
    ``os.environ`` is a HIGHER-precedence source, so it cannot defend against a
    leak that lives in ``os.environ`` -- which is where ``load_dotenv`` puts it.
    Pinned as a test because the whole design turns on it: had it been true,
    ~536 construction sites would have needed editing instead of one table.

    MUTATION that must make this fail: none in this repo -- this pins
    pydantic-settings' source ordering. It fails if a pydantic-settings upgrade
    ever ranks the dotenv file above the environment, at which point the comments
    in ``tests/_env_isolation.py`` need rewriting.
    """
    from trelix.core.config import EmbedderConfig

    monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "azure")
    # The ignore below is load-bearing, not laziness: `_env_file` is a real,
    # documented pydantic-settings runtime kwarg (BaseSettings.__init__ consumes
    # `_env_file` and friends), but it is absent from the generated __init__
    # signature mypy sees, so the checker rejects the very call this test exists
    # to characterise.
    assert EmbedderConfig(_env_file=None).provider == "azure"  # type: ignore[call-arg]


def test_litellm_dotenv_autoload_is_disabled() -> None:
    """Layer 2: litellm's own guard must evaluate to "skip load_dotenv".

    ``os.getenv("LITELLM_MODE", "DEV") == "DEV"`` is verbatim the condition in
    ``litellm/__init__.py`` that gates the ``load_dotenv`` call, so asserting the
    same expression is asserting the thing rather than a proxy for it. The
    literal ``"DEV"`` is written out, not imported from litellm.

    THIS is where the single-layer autoload mutant dies. With the scrub intact,
    making ``disable_litellm_dotenv_autoload`` a bare ``return`` leaves every
    provider assertion in ``test_operator_env_leak.py`` green -- the scrub cleans
    up after the injection -- so without this test that mutant would survive.

    MUTATION that must make this fail: make
    ``tests._env_isolation.disable_litellm_dotenv_autoload`` a bare ``return``,
    or change ``LITELLM_MODE_NON_DEV`` to ``"DEV"``.
    """
    assert os.getenv("LITELLM_MODE", "DEV") != "DEV"


def test_env_isolation_module_exposes_the_scrub() -> None:
    """``scrub_operator_env`` is public API of the shared isolation module.

    The unit conftest calls it; integration and eval deliberately do NOT (they
    need the operator's real credentials). A rename that only updated the unit
    conftest would leave that decision undocumented, so the name is pinned.

    MUTATION that must make this fail: rename ``scrub_operator_env``.
    """
    assert callable(_env_isolation.scrub_operator_env)
    assert callable(_env_isolation.apply_env_isolation)
