"""
The operator-env leak: importing ``litellm`` republishes the operator's dotenv
into ``os.environ``, so a unit test can silently exercise a DIFFERENT provider
than the one it claims to test.

MECHANISM, measured on this tree
--------------------------------
``litellm/__init__.py`` runs ``_dotenv.load_dotenv(override=False)`` at import
time when ``LITELLM_MODE`` is unset or ``DEV`` (its default). ``load_dotenv()``
with no argument calls ``find_dotenv()``, which walks up from the **caller's
frame directory** — here ``site-packages/litellm/`` — **not** from the process
cwd. So it climbs out of ``.venv`` and finds the ``.env`` of the repository that
OWNS the venv, from any cwd whatsoever.

That distinction matters because it widens the blast radius: it is not "pytest
run from inside this repository", it is *every* process using this venv.
Measured with two discriminating markers — a token present only in the real
repo's ``.env``, and a synthetic ``.env`` planted in the cwd — run from ``~``
with no ``.env`` in any ancestor, the repo's token still appeared and the
planted one did not.

The file it finds carries ``TRELIX_EMBEDDER_PROVIDER``, ``TRELIX_LLM_PROVIDER``
and the ``AZURE_*`` credentials. In one process, before the import
``EmbedderConfig().provider`` is ``"local"``; after it, ``"azure"``.

``src/trelix/core/config.py`` is NOT at fault. It already refuses to read a
cwd-relative ``.env`` (see ``resolve_operator_env_file``) precisely so a planted
file cannot be configuration, and its docstring's guarantee — "a dotenv key
never becomes a process environment variable, because nothing in trelix calls
``load_dotenv()``" — is true of trelix. It is a third-party import that breaks
it, through the one channel that anchoring cannot defend: ``os.environ`` itself.

WHY ``_env_file=None`` IS THE WRONG GUARD
-----------------------------------------
An earlier audit proposed passing ``_env_file=None`` at every construction site.
That does nothing here. pydantic-settings orders its sources
init > env > dotenv-file, so ``os.environ`` already outranks the file; removing
the file source leaves the leaked ``os.environ`` value in charge. Measured in
one process on this tree, after ``import litellm``:
``EmbedderConfig(_env_file=None).provider`` is ``"azure"``, not ``"local"``.
``tests/unit/test_env_isolation_covers_config_aliases.py`` pins that fact.

THIS IS A CORRECTNESS BUG, NOT A COST ONE
-----------------------------------------
The outbound call is already blocked: ``--disable-socket`` in ``addopts`` stops
the request before it leaves the process. Nothing is spent. What leaks is the
CONFIGURATION: a test named "local embedder does X" constructs an Azure
embedder and asserts on it.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

from trelix.core.config import EmbedderConfig, LLMConfig

# Module scope, deliberately: this reproduces the real trigger. The import is
# what mutates os.environ, and it must happen BEFORE the tests below run so the
# autouse isolation fixture is being asked to clean up after it rather than
# racing it. `importorskip` because litellm is an optional extra --
# tests/unit/test_retry.py gates on it the same way.
litellm = pytest.importorskip("litellm")


def test_litellm_import_is_the_live_trigger() -> None:
    """Precondition for every assertion in this file.

    Names the fixture that could make them vacuous: the module-scope
    ``pytest.importorskip("litellm")`` above. If litellm stops being installed,
    or the import is moved into a function, the leak channel is never opened and
    the assertions below would pass BY CONSTRUCTION while the hole is wide open.

    MUTATION that must make this fail: delete the module-scope
    ``pytest.importorskip("litellm")``.
    """
    assert "litellm" in sys.modules, (
        "module-scope pytest.importorskip('litellm') did not import litellm; "
        "the leak channel this file guards was never opened, so the other "
        "tests here prove nothing"
    )

    # `litellm in sys.modules` is NOT sufficient, and asserting only that made every
    # assertion in this file green-when-vacuous. Adversarial review measured it: with both
    # fix layers removed AND `dotenv.load_dotenv` stubbed to return False -- a faithful
    # stand-in for "no .env findable", i.e. CI in a container or a system-installed venv --
    # the file reported **9 passed** with the hole wide open, this precondition included.
    #
    # The real discriminating condition is that the channel CARRIED something: that
    # find_dotenv() resolved, from litellm's own frame, to a file that actually holds one of
    # the names we care about. Note find_dotenv walks up from the CALLER'S frame directory
    # (site-packages/litellm/), not the cwd, so this is asked of litellm's location.
    import dotenv

    dotenv_path = dotenv.find_dotenv(usecwd=False)
    if not dotenv_path or not pathlib.Path(dotenv_path).is_file():
        pytest.skip(
            "no .env is reachable from litellm's frame, so the leak channel cannot open "
            "here and these assertions would pass vacuously. This is the expected state in "
            "a container or a system-installed venv; it is NOT evidence the fix works -- "
            "test_scrub_removes_a_planted_provider_leak and "
            "test_litellm_dotenv_autoload_is_disabled are the self-contained proofs."
        )

    carried = set(dotenv.dotenv_values(dotenv_path))
    watched = {"TRELIX_EMBEDDER_PROVIDER", "TRELIX_LLM_PROVIDER", "AZURE_API_KEY", "AZURE_ENDPOINT"}
    # Names only, never values -- a failing `in os.environ` assertion makes pytest render
    # the whole environment into the CI log.
    assert carried & watched, (
        f"the .env at {dotenv_path} carries none of {sorted(watched)}, so importing litellm "
        "cannot demonstrate the leak and the assertions below prove nothing about it"
    )


def test_embedder_provider_is_local_after_litellm_import() -> None:
    """EmbedderConfig must still report the shipped default after the import.

    ``"local"`` is written as a literal on purpose: recomputing it from
    ``EmbedderConfig.model_fields`` would make this pass no matter what the
    environment said.

    MUTATION that must make this fail: BOTH layers at once -- make
    ``tests._env_isolation.disable_litellm_dotenv_autoload`` a bare ``return``
    AND set ``SCRUB_PREFIXES`` to ``()``. Measured: mutating either one alone
    leaves this GREEN, because either layer is on its own sufficient for the
    litellm channel. That is the point of the pair, and it is why this file is
    not where the single-layer mutants are killed -- see
    ``test_env_isolation_covers_config_aliases.py``, whose
    ``test_scrub_removes_a_planted_provider_leak`` plants its own leak (so it
    kills the scrub mutant with the autoload guard intact) and whose
    ``test_litellm_dotenv_autoload_is_disabled`` asserts litellm's own guard
    expression (so it kills the autoload mutant with the scrub intact).
    """
    assert EmbedderConfig().provider == "local"


def test_llm_provider_and_model_are_defaults_after_litellm_import() -> None:
    """Same for the chat/synthesis config, whose provider default differs.

    ``TRELIX_LLM_MODEL`` and ``TRELIX_LLM_PROVIDER`` were both unpinned by the
    old ~25-name isolation table, so this is a second, independent alias family
    the prefix scrub has to cover -- not a restatement of the embedder case.
    Both are present in this repository's own ``.env``, which is the file
    ``load_dotenv`` finds.

    MUTATION that must make this fail: as above, both layers at once.
    """
    cfg = LLMConfig()
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"


@pytest.mark.parametrize(
    "name",
    [
        "AZURE_API_KEY",
        "AZURE_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "TRELIX_EMBEDDER_PROVIDER",
        "TRELIX_LLM_PROVIDER",
    ],
)
def test_leaked_names_are_absent_from_process_env(name: str) -> None:
    """The credential names litellm republishes must not be visible to a test.

    Asserts the thing (the name is gone from ``os.environ``) rather than the
    proxy (a config field happens to look right), because a future config field
    reading one of these would otherwise pick it up unnoticed.

    MUTATION that must make this fail: both layers at once, for the reason given
    on ``test_embedder_provider_is_local_after_litellm_import``. Measured, so it
    is not a guess: with the autoload guard intact, deleting ``AZURE_API_KEY``
    from ``CONFIG_NON_PREFIXED_ENV`` leaves every case here GREEN and is killed
    only in ``test_env_isolation_covers_config_aliases.py`` (three tests there).

    Compared case-insensitively, matching pydantic-settings' own lookup
    (``case_sensitive`` defaults to False), and reduced to a list of NAMES
    before asserting: ``assert name not in os.environ`` makes pytest render the
    whole environment -- credential VALUES included -- into the failure report
    and the CI log.
    """
    present = sorted(k for k in os.environ if k.upper() == name)
    assert present == [], f"operator env leaked into the test process: {present}"
