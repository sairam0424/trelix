"""The GitHub App's child-process env allow-list must agree with what trelix reads.

The App (infra/github-app) is TypeScript; the names its `trelix index` / `trelix review`
children need live in Python (`config.py`, pinned here through `CONFIG_NON_PREFIXED_ENV`, and
the provider SDKs' own names, pinned through `INSTALLED_SDK_PROVIDER_ENV`). Neither language's
own suite ties the two together. Two mistakes would ship green:

* a provider name trelix reads is missing from `src/child-env.ts`, so the children run
  with the LLM or embedder silently unconfigured (`trelix review` then exits 3 and the PR
  gets a neutral "did not run" check); and
* a name the App must never share (its private key, its webhook secret, the platform's
  `RAILWAY_*` tokens, a GitHub token) is added to an allow-list, so an outside PR author's
  code can read it from a child's environment.

This reads the TypeScript source itself, the same way `test_github_app_env_contract.py`
reads the Dockerfile. It only looks at `const NAME: readonly string[] = [...]` literals;
a refactor that moves the lists out of that shape fails the first test on purpose, rather
than letting every check below pass over an empty extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._env_isolation import CONFIG_NON_PREFIXED_ENV, INSTALLED_SDK_PROVIDER_ENV
from trelix.core.config import WalkerConfig, resolve_operator_env_file

_CHILD_ENV_TS = (
    Path(__file__).resolve().parents[2] / "infra" / "github-app" / "src" / "child-env.ts"
)

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_ARRAY_LITERAL = re.compile(
    r"\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*readonly\s+string\[\]\s*=\s*\[(.*?)\]\s*;",
    re.DOTALL,
)
_STRING_LITERAL = re.compile(r'"([^"\\]*)"')
_FORCED_WALKER_FLAG = re.compile(r'\bTRELIX_WALKER_FOLLOW_SYMLINKS:\s*"([^"]*)"')
_SERVICE_ENV_PREFIX = re.compile(r'\bconst\s+SERVICE_ENV_PREFIX\s*=\s*"([^"]*)"')
_TRELIX_SOURCE = Path(__file__).resolve().parents[2] / "src" / "trelix"

_EXPECTED_ARRAYS = (
    "GIT_INHERITED_ENV",
    "TRELIX_INHERITED_ENV",
    "CONFIG_PROVIDER_ENV",
    "SDK_PROVIDER_ENV",
    "TRELIX_ENV_PREFIXES",
)

# Written out by hand, not derived: a secret of the App's own process, or of its platform,
# that no child may ever receive from the host. GITHUB_TOKEN is handed to the review child
# explicitly by `buildReviewChildEnv`, never copied from the host, so it is forbidden here too.
_FORBIDDEN_NAMES = (
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_WEBHOOK_SECRET",
    "GITHUB_APP_ID",
    "GITHUB_TOKEN",
    "RAILWAY_TOKEN",
    "RAILWAY_PROJECT_ID",
    "RAILWAY_ENVIRONMENT_ID",
    "RAILWAY_SERVICE_ID",
    "RAILWAY_PUBLIC_DOMAIN",
    "RAILWAY_API_TOKEN",
)

# Secret-looking names a deployment may hold for its own reasons; no allow-list may let one
# through. The TypeScript suite feeds the same names to the builders and requires every name
# in their output to come from an allow-list or a fixed name (child-env.test.ts, "closed world").
_UNRELATED_SECRET_NAMES = (
    "RAILWAY_API_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "STRIPE_SECRET_KEY",
    "AWS_SECRET_ACCESS_KEY_ID_2",
    "SOME_PASSWORD",
    "MY_API_KEY",
    "GH_TOKEN",
    "NPM_TOKEN",
    "SENTRY_DSN",
    "SLACK_BOT_TOKEN",
    "SENDGRID_API_KEY",
    "POSTGRES_PASSWORD",
    "JWT_SECRET",
    "SESSION_SECRET",
    "ENCRYPTION_KEY",
    "PRIVATE_KEY",
    "CLOUDFLARE_API_TOKEN",
    "VERCEL_TOKEN",
    "DOCKER_PASSWORD",
    "TWILIO_AUTH_TOKEN",
    "SSH_AUTH_SOCK",
)

# The provider names a deployment is known to set, so an emptied CONFIG_NON_PREFIXED_ENV
# cannot turn the set-equality check below into a vacuous pass.
_KNOWN_PROVIDER_NAMES = (
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_API_KEY",
    "AZURE_ENDPOINT",
    "OPENAI_API_KEY",
)

# Credential-chain names an operator on role-based AWS, Vertex, a gateway or Anthropic
# identity federation relies on. Written out by hand, so the equality check on
# SDK_PROVIDER_ENV below cannot be satisfied by an emptied INSTALLED_SDK_PROVIDER_ENV.
_KNOWN_SDK_CREDENTIAL_POINTERS = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_REGION",
    "AWS_ROLE_ARN",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_BASE_URL",
)

# The names a provider SDK reads that the children deliberately do NOT get, by reason. Every
# name in INSTALLED_SDK_PROVIDER_ENV that is neither in CONFIG_NON_PREFIXED_ENV nor here must
# be in SDK_PROVIDER_ENV, so a name added to the canonical list forces a decision: forward it
# or withhold it, with a reason. Where each is read was checked in the installed SDKs.
_WITHHELD_SDK_NAMES: dict[str, tuple[str, ...]] = {
    # They verify a webhook or administer an organisation; none of them calls a model.
    "verify or administer, no inference access": (
        "ANTHROPIC_ENVIRONMENT_KEY",
        "ANTHROPIC_WEBHOOK_SIGNING_KEY",
        "ANTHROPIC_WORK_SECRET",
        "OPENAI_ADMIN_KEY",
        "OPENAI_WEBHOOK_SECRET",
    ),
    # Read only by anthropic SDK client classes trelix never builds; it builds
    # anthropic.Anthropic (src/trelix/llm/providers/anthropic_backend.py).
    "anthropic platform clients trelix does not construct": (
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_GOOGLE_CLOUD_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
    ),
    # S3 and batch jobs, Azure services other than Azure OpenAI, Google search and SSO,
    # litellm's OpenTelemetry callback (trelix's own exporter reads
    # OTEL_EXPORTER_OTLP_ENDPOINT, which CONFIG_PROVIDER_ENV already carries).
    "services trelix has no client for": (
        "AWS_BATCH_ROLE_ARN",
        "AWS_S3_ENCRYPTION_KEY_ID",
        "AWS_S3_US_EAST_1_REGIONAL_ENDPOINT",
        "AWS_S3_USE_ARN_REGION",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
        "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION",
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_KEY_VAULT_URI",
        "AZURE_SENTINEL_CLIENT_SECRET",
        "AZURE_SENTINEL_ENDPOINT",
        "AZURE_SPEECH_API_BASE",
        "AZURE_SPEECH_API_KEY",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_CLIENT_SECRET",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_PSE_API_BASE",
        "GOOGLE_PSE_API_KEY",
        "OTEL_ENDPOINT",
    ),
    # A service principal is a tenant-wide identity, not an LLM endpoint credential. The
    # chain is also unusable here: AZURE_CLIENT_ID and AZURE_TENANT_ID are not in the
    # canonical list, and trelix builds AzureOpenAI with an explicit api key.
    "azure service-principal secrets": (
        "AZURE_CERTIFICATE_PASSWORD",
        "AZURE_CERTIFICATE_PATH",
        "AZURE_CLIENT_SECRET",
        "AZURE_CREDENTIAL",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_PASSWORD",
    ),
}


def _child_env_source() -> str:
    """child-env.ts without its comments, which quote the very shapes searched for below."""
    source = _CHILD_ENV_TS.read_text(encoding="utf-8")
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", source))


def _allow_lists() -> dict[str, list[str]]:
    """Every `const NAME: readonly string[] = [...]` in child-env.ts, by name."""
    return {
        name: _STRING_LITERAL.findall(body)
        for name, body in _ARRAY_LITERAL.findall(_child_env_source())
    }


def _covers(entry: str, name: str) -> bool:
    """True when the allow-list entry lets `name` through (an exact match or a `FOO_` prefix)."""
    return entry == name or (entry.endswith("_") and name.startswith(entry))


def test_the_allow_lists_are_extracted() -> None:
    """Precondition for every other test here: the five lists exist and are not empty."""
    lists = _allow_lists()

    assert sorted(lists) == sorted(_EXPECTED_ARRAYS)
    for name, entries in lists.items():
        assert entries, f"{name} is empty or no longer a string-literal array"
        assert len(entries) == len(set(entries)), f"{name} lists a name twice"


@pytest.mark.parametrize("forbidden", _FORBIDDEN_NAMES)
def test_no_allow_list_lets_a_forbidden_name_through(forbidden: str) -> None:
    offenders = {
        array: [entry for entry in entries if _covers(entry, forbidden)]
        for array, entries in _allow_lists().items()
    }

    assert {array: hits for array, hits in offenders.items() if hits} == {}


@pytest.mark.parametrize("unrelated", _UNRELATED_SECRET_NAMES)
def test_no_allow_list_lets_an_unrelated_secret_name_through(unrelated: str) -> None:
    offenders = {
        array: [entry for entry in entries if _covers(entry, unrelated)]
        for array, entries in _allow_lists().items()
    }

    assert {array: hits for array, hits in offenders.items() if hits} == {}


def test_the_only_prefix_is_trelix() -> None:
    """A prefix passes every name under it, so a second one needs a reviewer's eyes."""
    assert _allow_lists()["TRELIX_ENV_PREFIXES"] == ["TRELIX_"]


def test_every_non_prefixed_name_trelix_reads_is_allow_listed() -> None:
    """Both directions: nothing config.py reads is missing, nothing it stopped reading lingers."""
    allowed = set(_allow_lists()["CONFIG_PROVIDER_ENV"])

    assert set(CONFIG_NON_PREFIXED_ENV) - allowed == set(), "trelix reads it, the children lose it"
    assert allowed - set(CONFIG_NON_PREFIXED_ENV) == set(), (
        "allow-listed but trelix no longer reads it"
    )


def test_the_known_provider_names_are_allow_listed() -> None:
    """Control for the test above: it is not green only because both sides came up empty."""
    allowed = set(_allow_lists()["CONFIG_PROVIDER_ENV"])

    assert set(_KNOWN_PROVIDER_NAMES) <= allowed
    assert set(_KNOWN_PROVIDER_NAMES) <= set(CONFIG_NON_PREFIXED_ENV)


def test_the_sdk_names_are_the_installed_provider_names_less_config_and_withheld() -> None:
    """Both directions, so a typo, a lost credential chain or a new SDK name is a failure.

    A typo would allow-list a name nothing reads and leave the real one withheld. A name
    missing from the list is an operator on Vertex or role-based AWS losing authentication
    (`trelix review` exits 3 and the PR gets a neutral check).
    """
    sdk_names = set(_allow_lists()["SDK_PROVIDER_ENV"])
    withheld = {name for names in _WITHHELD_SDK_NAMES.values() for name in names}
    expected = set(INSTALLED_SDK_PROVIDER_ENV) - set(CONFIG_NON_PREFIXED_ENV) - withheld

    assert sdk_names - expected == set(), "allow-listed but not a name an installed SDK reads"
    assert expected - sdk_names == set(), "an SDK reads it, the children lose it"


def test_the_withheld_sdk_names_are_real_and_not_also_forwarded() -> None:
    """No stale entry in the withheld table, and nothing withheld is quietly allow-listed."""
    withheld = {name for names in _WITHHELD_SDK_NAMES.values() for name in names}
    allowed = {entry for entries in _allow_lists().values() for entry in entries}

    assert withheld - set(INSTALLED_SDK_PROVIDER_ENV) == set(), "withheld but no longer listed"
    assert withheld & set(CONFIG_NON_PREFIXED_ENV) == set(), "config.py reads it; cannot withhold"
    assert withheld & allowed == set(), "withheld on purpose yet allow-listed"


def test_the_known_sdk_credential_pointers_are_allow_listed() -> None:
    """Control for the equality test above: it is not green only because both sides are empty."""
    allowed = set(_allow_lists()["SDK_PROVIDER_ENV"])

    assert set(_KNOWN_SDK_CREDENTIAL_POINTERS) <= allowed
    assert set(_KNOWN_SDK_CREDENTIAL_POINTERS) <= set(INSTALLED_SDK_PROVIDER_ENV)


def test_the_inherited_lists_are_exactly_what_each_child_needs() -> None:
    """git gets PATH and LANG only; the trelix children also get HOME and the config home."""
    lists = _allow_lists()

    assert lists["GIT_INHERITED_ENV"] == ["PATH", "LANG"]
    assert lists["TRELIX_INHERITED_ENV"] == ["PATH", "LANG", "HOME", "XDG_CONFIG_HOME"]


def test_the_config_home_variable_the_children_inherit_is_the_one_trelix_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """trelix finds the operator's env file through XDG_CONFIG_HOME, so a child must keep it."""
    config_home = tmp_path / "xdg"
    (config_home / "trelix").mkdir(parents=True)
    operator_file = config_home / "trelix" / "env"
    operator_file.write_text("", encoding="utf-8")
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.delenv("TRELIX_CONFIG_FILE", raising=False)
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    assert resolve_operator_env_file() == operator_file.resolve()
    assert "XDG_CONFIG_HOME" in _allow_lists()["TRELIX_INHERITED_ENV"]


def test_the_apps_own_settings_prefix_is_withheld_and_trelix_reads_nothing_under_it() -> None:
    """TRELIX_APP_* configure the service (kill switch, install allow-list, queue sizes).

    child-env.ts keeps them out of both children, because the review child reads text an
    outside author wrote. That is only safe while trelix itself reads no name under the
    prefix: a setting that reused it would silently stop reaching the children.
    """
    assert _SERVICE_ENV_PREFIX.findall(_child_env_source()) == ["TRELIX_APP_"]

    offenders = sorted(
        str(path.relative_to(_TRELIX_SOURCE))
        for path in _TRELIX_SOURCE.rglob("*.py")
        if "TRELIX_APP_" in path.read_text(encoding="utf-8", errors="replace")
    )

    assert offenders == []
    assert any(_TRELIX_SOURCE.rglob("*.py")), "the scan must have looked at trelix's source"


def test_the_forced_walker_flag_is_a_name_and_value_trelix_honours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """child-env.ts forces the walker off for both trelix children: prove trelix reads it so."""
    forced = _FORCED_WALKER_FLAG.findall(_child_env_source())
    assert len(forced) == 1, "child-env.ts must force TRELIX_WALKER_FOLLOW_SYMLINKS exactly once"

    monkeypatch.setenv("TRELIX_WALKER_FOLLOW_SYMLINKS", forced[0])

    assert WalkerConfig().follow_symlinks is False
