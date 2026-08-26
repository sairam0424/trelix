"""
Shared env isolation for the unit, integration and eval suites.

Beast-mode feature flags default to False in code, but a developer's ``.env``
may have them set to True for local testing. Every suite has to override them
so tests observe the shipped code defaults regardless of ``.env``.

WHY THIS IS ONE MODULE
----------------------
This table used to be copy-pasted into ``tests/{unit,integration,eval}/
conftest.py``. The copies drifted: only the unit copy grew ``_CODE_DEFAULTS``
and ``_ENV_PREFIXES_TO_SCRUB`` when scalar compression tuning landed, so a
developer with ``TRELIX_RETRIEVAL_COMPRESSION_RATIO=0.2`` in ``.env`` got that
value inside integration and eval runs while unit runs correctly saw 0.45.
That is the exact contamination the eval conftest calls "load-bearing": the
compression A/B scores faithfulness against the assembled context, so an "off"
baseline built with a dev's ratio is not comparable to CI's. One table means a
new setting cannot be pinned in one suite and forgotten in the other two.

The tables are ``MappingProxyType``/tuples because three conftests now share
them: an in-place edit from one suite's fixture would silently change what the
other two observe.

pydantic-settings reads env vars BEFORE the .env file, so setenv("X", "false")
overrides whatever .env contains for that key. RetrievalConfig uses
``extra="ignore"``, so neutralizing a compression flag that hasn't landed as a
field yet is harmless (ignored now, auto-neutralized once the field exists).

SCOPE LIMIT — read this before adding isolation here
----------------------------------------------------
``monkeypatch`` is IN-PROCESS ONLY. Nothing here affects a `trelix` binary
spawned via ``subprocess``: the child re-reads the process environment *and*
the ``.env`` file in its own cwd from scratch. Tests that shell out
(``tests/integration/test_cli.py``) must isolate their child explicitly — that
file scrubs ``TRELIX_*`` out of the child env and runs it from a directory with
no ``.env``. Adding a var here will NOT fix a subprocess test.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType

import pytest

# Boolean beast-mode flags. Union of what the three suites pinned separately;
# the compression aliases carry both plausible spellings so the off-baseline
# stays byte-identical no matter which the field ultimately binds to.
BEAST_MODE_DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        # Context compression (A/B isolation -- the load-bearing entry here).
        "TRELIX_RETRIEVAL_COMPRESSION": "false",
        "TRELIX_RETRIEVAL_COMPRESSION_ENABLED": "false",
        # Other beast-mode retrieval flags.
        "TRELIX_FILE_SUMMARIES_ENABLED": "false",
        "TRELIX_RETRIEVAL_FILE_SUMMARY_LEG": "false",
        "TRELIX_RETRIEVAL_HYDE_FALLBACK": "false",
        "TRELIX_RETRIEVAL_MULTI_QUERY": "false",
        "TRELIX_RETRIEVAL_FLARE": "false",
        "TRELIX_RETRIEVAL_PAGERANK_BOOST": "false",
        "TRELIX_TELEMETRY_ENABLED": "false",
        "TRELIX_OTEL_ENABLED": "false",
    }
)

# Non-flag settings pinned to their CODE defaults. Same rationale as
# BEAST_MODE_DEFAULTS, but these carry a value rather than a boolean, so they
# must be pinned to the default itself instead of "false" — a developer's .env
# tuning compression for a local experiment must not change what the
# "unconfigured" tests observe.
CODE_DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        "TRELIX_RETRIEVAL_COMPRESSION_PROVIDER": "extractive",
        "TRELIX_RETRIEVAL_COMPRESSION_RATIO": "0.45",
        "TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS": "120",
    }
)

# Per-intent override families read straight from os.environ (NOT via
# pydantic-settings), so pinning the scalar above does not neutralize them —
# every var under these prefixes has to be removed by name. The trailing
# underscore is deliberate: it matches the per-intent suffixes only, never the
# bare scalar that CODE_DEFAULTS pins.
ENV_PREFIXES_TO_SCRUB: tuple[str, ...] = ("TRELIX_RETRIEVAL_COMPRESSION_RATIO_",)

# Non-boolean settings that must stay unset (not "false") for tests to see the
# "unconfigured" code default — same rationale as BEAST_MODE_DEFAULTS above,
# but these aren't flags, so setenv("...", "false") would misconfigure them
# instead of disabling them. delenv() alone is NOT enough here: it only
# removes the var from the process environment, but pydantic-settings'
# .env-file source reads the file directly and independently — if a
# developer's .env has a real value for one of these (e.g. a live
# TRELIX_LINEAR_API_KEY for manual connector testing), removing the
# process-env copy doesn't stop the file fallback from supplying it
# anyway. Overriding to "" (falsy, same as unset for every `if not val`
# validate_config() check in this codebase) is what actually neutralizes
# the .env value for the test process.
UNSET_BY_DEFAULT: tuple[str, ...] = ("TRELIX_API_AUTH_TOKEN",)
EMPTY_STRING_BY_DEFAULT: tuple[str, ...] = (
    "TRELIX_LINEAR_API_KEY",
    "TRELIX_LINEAR_TEAM_KEY",
    "TRELIX_JIRA_BASE_URL",
    "TRELIX_JIRA_EMAIL",
    "TRELIX_JIRA_API_TOKEN",
    "TRELIX_JIRA_PROJECT_KEY",
    "TRELIX_TESTRAIL_BASE_URL",
    "TRELIX_TESTRAIL_USERNAME",
    "TRELIX_TESTRAIL_API_KEY",
)
# Int-typed connector fields (project_id: int | None) can't take the ""
# override above — pydantic would fail to parse "" as an int the same way
# it fails on a malformed real value. "0" parses fine and is falsy, so
# every `if not val` validate_config() check in this codebase still treats
# it as "missing" — same effective behavior as the string fields above.
ZERO_INT_BY_DEFAULT: tuple[str, ...] = ("TRELIX_TESTRAIL_PROJECT_ID",)


# ---------------------------------------------------------------------------
# Deny-by-default operator-env scrub (unit suite only)
# ---------------------------------------------------------------------------
#
# THE LEAK THIS CLOSES, measured on this tree. ``import litellm`` runs
# ``dotenv.load_dotenv(override=False)`` at module import time (its ``DEV``
# default). CORRECTED MECHANISM, and the blast radius is wider than it looks:
# ``find_dotenv()`` walks up from the CALLER'S FRAME directory -- here
# ``site-packages/litellm/`` -- not from the process cwd. So it climbs out of
# ``.venv`` and finds the ``.env`` of the repository that OWNS the venv, from any
# cwd whatsoever. Measured with two discriminating markers (a token present only
# in the real repo's ``.env``, and a synthetic ``.env`` planted in the cwd): run
# from ``~`` with no ``.env`` in any ancestor, the repo's token still appears and
# the planted one does not. Every process using this venv leaks, not merely ones
# started inside the repository. In one process:
# ``EmbedderConfig().provider`` is ``"local"`` before the import and ``"azure"``
# after it. That is a CORRECTNESS failure, not a cost one — ``--disable-socket``
# already stops the outbound call — because a test named for the local embedder
# silently constructs and asserts on an Azure one.
#
# config.py is not at fault: it already refuses to read a cwd-relative ``.env``
# (``resolve_operator_env_file``). The leak arrives through ``os.environ``, the
# one channel that anchoring cannot defend.
#
# WHY A PREFIX SCRUB AND NOT A LONGER TABLE OF NAMES. The tables above pin ~25
# names. config.py's eighteen settings classes read 273 distinct env names, most
# of which have no ``alias=`` to grep for at all because they are
# ``env_prefix`` + field name. Enumerating them by hand guarantees the table
# lags the next field someone adds — which is exactly how ~80 names, including
# ``TRELIX_LLM_PROVIDER`` and ``TRELIX_LLM_MODEL``, came to be unpinned. A
# prefix denies the whole namespace, including names that do not exist yet.
#
# WHY ``_env_file=None`` IS NOT THE FIX. It removes pydantic-settings' dotenv
# FILE source, but ``os.environ`` is a higher-precedence source, so it cannot
# defend against a value that has been leaked INTO ``os.environ``. Measured:
# after ``import litellm``, ``EmbedderConfig(_env_file=None).provider`` is still
# ``"azure"``. ``tests/unit/test_env_isolation_covers_config_aliases.py`` pins
# that, because the alternative design (editing hundreds of construction sites)
# rests on it being false.
SCRUB_PREFIXES: tuple[str, ...] = ("TRELIX_",)

# The env names config.py reads that are NOT under a scrubbed prefix: provider
# SDK conventions an operator has set for reasons unrelated to trelix. Held as an
# explicit table rather than a second prefix rule because ``AZURE_``/``AWS_``/
# ``GOOGLE_`` are not trelix's namespace — blanket-scrubbing them would reach
# past config.py's surface into whatever else the developer's shell needs.
#
# Kept honest by ``tests/unit/test_env_isolation_covers_config_aliases.py``,
# which derives the same selection from config.py's model fields at runtime and
# fails set-equality in BOTH directions: a new non-prefixed alias in config.py
# is an uncovered-name failure, and an entry here for a name config.py stopped
# reading is a stale-entry failure.
CONFIG_NON_PREFIXED_ENV: tuple[str, ...] = (
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
)

# ---------------------------------------------------------------------------
# HOLE 1, CLOSED: names an installed SDK reads that config.py never declares
# ---------------------------------------------------------------------------
#
# ``CONFIG_NON_PREFIXED_ENV`` above is scoped to what config.py declares, and its
# meta-test derives the same selection from config.py's model fields. That guard
# is honest about drift in config.py and BLIND to the other half of the channel:
# a provider SDK reads its OWN env names straight out of ``os.environ``, with no
# pydantic field anywhere in trelix to derive them from. Round 4 established the
# case that makes this concrete -- "Azure credentials are read by the OpenAI SDK
# directly from os.environ, not via pydantic env_prefix" -- so the config-derived
# meta-test can be GREEN while a real channel is open. That is a green-when-
# vacuous shape in the guard itself, and this table is what closes it.
#
# THE THREE NAMES THE ROUND-4 AUTHOR CALLED OUT BY HAND, and where they are read
# in the INSTALLED packages of this venv (grepped, not remembered):
#   AWS_SESSION_TOKEN   botocore/credentials.py:1192  (EnvProvider.TOKENS)
#   AWS_DEFAULT_REGION  botocore/configprovider.py:74 (session var 'region')
#   OPENAI_BASE_URL     openai/_client.py:251         (base_url default)
# None of the three has a config.py field, so none could ever have been derived.
# AWS_SESSION_TOKEN is the sharpest illustration of why a grep for
# ``os.environ.get("NAME")`` is not enough on its own: botocore never writes that
# call: the name lives in a class-attribute list which ``EnvProvider`` indexes at
# runtime. A derivation that only matched direct-call sites would have missed the
# very name the hole was named after, which is why the meta-test's scanner
# matches quoted string LITERALS and filters them by shape (see
# ``tests/unit/test_env_isolation_covers_sdk_env.py``, which also states the
# limits of that approach).
#
# WHY A SEPARATE TUPLE RATHER THAN EXTENDING CONFIG_NON_PREFIXED_ENV. That table
# is pinned by set-equality in BOTH directions against config.py's derived names
# (``test_non_prefixed_table_has_no_stale_entries``). Adding AWS_SESSION_TOKEN to
# it would be a stale entry by that test's definition -- correctly so, because
# config.py really does not read it. Two tables with two derivations keeps both
# guards strict instead of loosening one to fit the other.
#
# DO ENDPOINTS AND REGIONS BELONG IN THE SAME TABLE AS SECRETS? Yes, and the
# argument is not "they are also sensitive" -- OPENAI_BASE_URL and
# AWS_DEFAULT_REGION are not credentials, and a leaked region is not a
# disclosure. The argument is that this table's axis was never secrecy. Round 4
# recorded the leak as "a CORRECTNESS failure, not a cost one ... a test named for
# the local embedder silently constructs and asserts on an Azure one", and
# ``--disable-socket`` already removes the spend and exfiltration risk. The axis
# is therefore: DOES THIS NAME CHANGE WHICH PROVIDER OR ENDPOINT THE CODE UNDER
# TEST RESOLVES? A base_url does exactly that -- it redirects every request the
# constructed client would make, so a test asserting on "the default OpenAI
# client" is asserting on the operator's gateway instead. A region does it more
# weakly but really: for Bedrock it is a component of the endpoint host and it
# selects the model-availability surface. Both change the answer to "which
# provider did this test actually exercise", so both belong.
#
# What that same axis EXCLUDES is the reason this is a family-filtered table and
# not a second blanket prefix rule: a name that only affects the operator's
# unrelated tooling has no bearing on which provider trelix resolves, so
# scrubbing all of ``AWS_*`` would reach past the question being asked. The
# families here are trelix's own provider surface, taken from
# CONFIG_NON_PREFIXED_ENV.
#
# Over-inclusion within a family is deliberate and safe. Selection is by NAME
# SHAPE, so a boolean like AWS_USE_FIPS_ENDPOINT is pulled in even though it is
# neither a credential nor an endpoint. Deny-by-default plus monkeypatch teardown
# makes a scrubbed-but-harmless name a non-event; an UNSCRUBBED credential is the
# failure that matters. Verified not to break the suite rather than assumed --
# see the deliverable's proof runs.
INSTALLED_SDK_PROVIDER_ENV: tuple[str, ...] = (
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_AWS_API_BASE",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_AWS_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "ANTHROPIC_ENVIRONMENT_KEY",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_WEBHOOK_SIGNING_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_ACCOUNT_ID",
    "AWS_BATCH_ROLE_ARN",
    "AWS_BEDROCK_BASE_URL",
    "AWS_BEDROCK_RUNTIME_ENDPOINT",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_ENDPOINT_URL",
    "AWS_PROFILE",
    "AWS_PROFILE_NAME",
    "AWS_REGION",
    "AWS_REGION_NAME",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_S3_ENCRYPTION_KEY_ID",
    "AWS_S3_US_EAST_1_REGIONAL_ENDPOINT",
    "AWS_S3_USE_ARN_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_NAME",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_USE_DUALSTACK_ENDPOINT",
    "AWS_USE_FIPS_ENDPOINT",
    "AWS_WEB_IDENTITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AZURE_AD_TOKEN",
    "AZURE_AI_API_BASE",
    "AZURE_AI_API_KEY",
    "AZURE_AI_API_VERSION",
    "AZURE_API_BASE",
    "AZURE_API_KEY",
    "AZURE_API_VERSION",
    "AZURE_CERTIFICATE_PASSWORD",
    "AZURE_CERTIFICATE_PATH",
    "AZURE_CLIENT_SECRET",
    "AZURE_CREDENTIAL",
    "AZURE_DEFAULT_RESPONSES_API_VERSION",
    "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION",
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_KEY_VAULT_URI",
    "AZURE_OPENAI_AD_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_PASSWORD",
    "AZURE_SENTINEL_CLIENT_SECRET",
    "AZURE_SENTINEL_ENDPOINT",
    "AZURE_SPEECH_API_BASE",
    "AZURE_SPEECH_API_KEY",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "AZURE_STORAGE_CLIENT_SECRET",
    "CO_API_KEY",
    "CO_API_URL",
    "COHERE_API_BASE",
    "COHERE_API_KEY",
    "GEMINI_API_BASE",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_PSE_API_BASE",
    "GOOGLE_PSE_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_API_VERSION",
    "OPENAI_BASE_URL",
    "OPENAI_CHATGPT_API_BASE",
    "OPENAI_LIKE_API_BASE",
    "OPENAI_LIKE_API_KEY",
    "OPENAI_WEBHOOK_SECRET",
    "OTEL_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "QDRANT_API_BASE",
    "QDRANT_API_KEY",
    "QDRANT_URL",
    "VERTEX_API_BASE",
    "VERTEX_CREDENTIALS",
    "VERTEXAI_API_BASE",
    "VERTEXAI_CREDENTIALS",
    "VOYAGE_AI_API_KEY",
    "VOYAGE_AI_TOKEN",
    "VOYAGE_API_BASE",
    "VOYAGE_API_KEY",
    "VOYAGE_API_KEY_PATH",
)


# Set once, at import: litellm reads it at ITS import time, so a fixture would
# run far too late — same reasoning as HF_HUB_OFFLINE in tests/unit/conftest.py.
LITELLM_MODE_ENV_VAR = "LITELLM_MODE"
LITELLM_MODE_NON_DEV = "PRODUCTION"


def disable_litellm_dotenv_autoload() -> None:
    """Stop the leak at source: make ``import litellm`` skip ``load_dotenv()``.

    litellm guards that call with ``if os.getenv("LITELLM_MODE", "DEV") ==
    "DEV"``, so any other value skips it. Audited on the pinned version: the
    only other reader of the flag is ``litellm/proxy/proxy_cli.py``, which this
    suite never invokes, so nothing else in litellm's behaviour changes.

    WHY THIS AND NOT A no-op PATCH OF ``dotenv.load_dotenv``. That was the first
    design and it is wrong here: ``tests/perf/test_query_latency.py`` calls
    ``load_dotenv()`` deliberately (anchored at the repo root, on purpose), and
    ``testpaths = ["tests"]`` means a bare ``pytest`` run collects it into the
    SAME process. Neutralising the library globally would fix the unit suite by
    silently breaking the perf suite — the same shape of mistake as applying one
    provider's fix to all three.

    Assigned unconditionally rather than via ``setdefault``, which is where this
    differs from its ``HF_HUB_OFFLINE`` neighbour. Offline mode has a legitimate
    override (a developer debugging a genuine Hub fetch), whereas there is no
    version of a unit test that wants a dotenv file published into its process
    environment. ``scrub_operator_env`` is the guard that must hold even if this
    one is defeated; this is the narrower belt that stops the pollution instead
    of cleaning it up afterwards.
    """
    os.environ[LITELLM_MODE_ENV_VAR] = LITELLM_MODE_NON_DEV


def scrub_operator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every operator-supplied config input from the process environment.

    Deny-by-default: pydantic then falls back to each field's own declared
    default, which is what a unit test asserting on "the unconfigured behaviour"
    means. Undone at teardown by ``monkeypatch``, so a test that wants a value
    can still set one in its own body.

    Matched case-INSENSITIVELY, because pydantic-settings' ``case_sensitive``
    defaults to False: it would resolve a lowercase ``azure_api_key`` in the
    environment, so a case-sensitive scrub would leave a live channel open.

    Call this BEFORE ``apply_env_isolation``: the pins that function applies are
    ``TRELIX_*`` names and a scrub running afterwards would delete them again.

    Two name tables, unioned: ``CONFIG_NON_PREFIXED_ENV`` (what config.py declares)
    and ``INSTALLED_SDK_PROVIDER_ENV`` (what an installed provider SDK reads for
    itself, which config.py cannot be asked about). They overlap -- OPENAI_API_KEY is
    in both -- and the union is deliberate rather than a hierarchy: each is pinned by
    its own derivation, and neither is authoritative over the other.
    """
    targets = {name.upper() for name in CONFIG_NON_PREFIXED_ENV}
    targets |= {name.upper() for name in INSTALLED_SDK_PROVIDER_ENV}
    # list(...) first: delenv mutates os.environ while we iterate it.
    for name in list(os.environ):
        upper = name.upper()
        if upper.startswith(SCRUB_PREFIXES) or upper in targets:
            monkeypatch.delenv(name, raising=False)


def apply_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin beast-mode flags and connector settings to their code defaults.

    Called from each suite's autouse ``_isolate_beast_mode_flags`` fixture, so
    every override is undone at teardown by ``monkeypatch``.
    """
    for var, val in BEAST_MODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
    for var, val in CODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
    for prefix in ENV_PREFIXES_TO_SCRUB:
        for var in [k for k in os.environ if k.startswith(prefix)]:
            monkeypatch.delenv(var, raising=False)
    for var in UNSET_BY_DEFAULT:
        monkeypatch.delenv(var, raising=False)
    for var in EMPTY_STRING_BY_DEFAULT:
        monkeypatch.setenv(var, "")
    for var in ZERO_INT_BY_DEFAULT:
        monkeypatch.setenv(var, "0")
