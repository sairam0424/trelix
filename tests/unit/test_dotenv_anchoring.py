"""SEC-03a — a repo-local ``.env`` is not a configuration source.

Every settings model in this project used to declare ``env_file=".env"``, which
pydantic-settings resolves against the **process cwd**. trelix's cwd is routinely
inside a repository trelix does not own: ``trelix index .`` in the PR-checkout of
``.github/workflows/trelix-review.yml``, ``trelix review`` on a fork branch, a
cloned dependency. So a ``.env`` committed by whoever wrote that repository was a
live configuration source for all fifteen models — able to repoint providers,
endpoints and credential fields, and (via ``local_code_model`` /
``nomic_code_model``) to name the model whose Python
``SentenceTransformer(..., trust_remote_code=True)`` executes in this process.

The tests below are written from the attacker's side: they plant the hostile
``.env`` and assert the values are **refused**, not merely that a legitimate
configuration still loads. The two "still honoured" cases exist so the fix cannot
degenerate into deleting dotenv support — an operator-owned file must still work.

Subprocess probes are used where the anchor itself is under test: the anchor is
resolved once at import (see ``core/config.resolve_operator_env_file``), and
``monkeypatch`` cannot un-import a module. The child gets an env built from
scratch — no ``TRELIX_*`` / ``OPENAI_*`` / ``AZURE_*`` inherited — plus its own
``HOME``, which is exactly the "clean scratch dir" the audit reproduced from.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

from trelix.core.config import EmbedderConfig, resolve_operator_env_file

# A value no operator would ever configure, so a hostile-.env hit is unambiguous.
HOSTILE_MODEL = "attacker.example/pwn-model"
OPERATOR_MODEL = "operator.example/allowed-model"

_HOSTILE_DOTENV = (
    f"TRELIX_EMBEDDER_PROVIDER=nomic-code\n"
    f"TRELIX_EMBEDDER_LOCAL_CODE_MODEL={HOSTILE_MODEL}\n"
    f"TRELIX_EMBEDDER_NOMIC_CODE_MODEL={HOSTILE_MODEL}\n"
    f"TRELIX_LLM_MODEL={HOSTILE_MODEL}\n"
    f"AZURE_ENDPOINT=https://attacker.example/v1\n"
    f"TRELIX_API_AUTH_TOKEN=attacker-chosen-token\n"
)

# Reads the two fields the audit measured moving under a planted .env.
_PROBE = (
    "import json;"
    "from trelix.core.config import EmbedderConfig;"
    "c = EmbedderConfig();"
    "print(json.dumps({'provider': c.provider, 'local_code_model': c.local_code_model}))"
)


def _field_default(name: str) -> object:
    """The pristine, unconfigured value — read off the model, not hardcoded."""
    return EmbedderConfig.model_fields[name].default


def _write_mutmut_bootstrap_if_needed(cwd: Path) -> None:
    """Neutralise a mutmut-only crash in the child, without touching what is under test.

    Under `scripts/mutation.py`, any scope inside the eager `trelix` import graph
    (indexing.walker, graph, store.vector, store.db, retrieval.*) makes `import
    trelix` pull in a module wrapped by mutmut's own trampoline
    (`mutmut/mutation/trampoline.py`), which imports `mutmut.__main__`, which at
    MODULE SCOPE (`mutmut/utils/safe_setproctitle.py:15`, unconditional, not gated
    on `MUTANT_UNDER_TEST`) calls `Config.get()`. That reads `pyproject.toml`
    relative to `os.getcwd()` and, finding none, falls back to
    `_guess_source_paths()` (also `os.getcwd()`-relative), which raises
    `FileNotFoundError: Could not figure out where the code to mutate is` in ANY
    process whose cwd has no `pyproject.toml`/`lib`/`src` — which the probe's own
    `cwd` deliberately is, by design (see module docstring).

    `cwd` must stay the untrusted repo: that is the fixture under test. So this
    writes a `pyproject.toml` naming a `[tool.mutmut]` `source_paths` INTO that
    same directory instead, purely to satisfy mutmut's own bootstrap. trelix's own
    config resolution never reads `pyproject.toml` (see
    `core.config.resolve_operator_env_file`), so this cannot change what the test
    asserts. It is a no-op file that only matters when mutmut is already loaded in
    THIS process (i.e. only under `scripts/mutation.py`, never in a normal test run
    or in CI, where "mutmut" is never imported and this function does nothing).
    """
    if "mutmut" not in sys.modules:
        return
    marker = cwd / "pyproject.toml"
    if marker.exists():
        return
    marker.write_text('[tool.mutmut]\nsource_paths = ["."]\n', encoding="utf-8")


def _probe_config(*, cwd: Path, home: Path, extra_env: dict[str, str] | None = None) -> dict:
    """Construct EmbedderConfig in a child process with a from-scratch environment."""
    _write_mutmut_bootstrap_if_needed(cwd)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        # The child must import the trelix under test, not an installed copy.
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        **(extra_env or {}),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture
def hostile_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory that is only ever *indexed* — its .env is attacker-controlled."""
    repo = tmp_path / "untrusted-repo"
    repo.mkdir()
    (repo / ".env").write_text(_HOSTILE_DOTENV, encoding="utf-8")
    monkeypatch.chdir(repo)
    for var in list(os.environ):
        if var.startswith(("TRELIX_", "OPENAI_", "AZURE_")):
            monkeypatch.delenv(var, raising=False)
    return repo


class TestHostileDotenvIsNotConfiguration:
    def test_embedder_provider_and_model_stay_pristine(self, hostile_repo: Path) -> None:
        config = EmbedderConfig()

        assert config.provider != "nomic-code"
        assert config.local_code_model != HOSTILE_MODEL
        assert config.nomic_code_model != HOSTILE_MODEL

    def test_llm_endpoint_is_not_redirected(self, hostile_repo: Path) -> None:
        """The exfiltration shape: repoint the synthesis endpoint at the attacker."""
        from trelix.core.config import LLMConfig

        config = LLMConfig()

        assert config.model != HOSTILE_MODEL
        assert config.azure_endpoint != "https://attacker.example/v1"

    def test_api_auth_token_is_not_attacker_chosen(self, hostile_repo: Path) -> None:
        """app.py declares the fifteenth env_file — a fix that missed it is a site-fix.

        An attacker-known token is worse than no token: on a deployment that left
        auth off it fabricates a credential the attacker holds, and on one that set
        a token it replaces the operator's.
        """
        pytest.importorskip("fastapi")
        from trelix.api.app import _ApiAuthSettings

        assert _ApiAuthSettings().api_auth_token != "attacker-chosen-token"

    def test_no_settings_model_declares_a_cwd_relative_env_file(self) -> None:
        """The anti-site-fix guard: fifteen models declared it, so count them all.

        A relative ``env_file`` is resolved against whatever directory the process
        happens to be in. Any new model that re-introduces one fails here.
        """
        pytest.importorskip("fastapi")
        import trelix.api.app as app_module
        import trelix.core.config as config_module

        offenders: list[str] = []
        for module in (config_module, app_module):
            for name, obj in vars(module).items():
                if not (isinstance(obj, type) and issubclass(obj, BaseSettings)):
                    continue
                declared = obj.model_config.get("env_file")
                if declared is None:
                    continue
                for entry in declared if isinstance(declared, list | tuple) else [declared]:
                    if not Path(entry).is_absolute():
                        offenders.append(f"{module.__name__}.{name} -> {entry!r}")

        assert not offenders, "cwd-relative env_file declarations: " + ", ".join(offenders)


class TestAnchorInAChildProcess:
    """The audit's own reproduction: a clean scratch dir holding only a .env."""

    def test_hostile_dotenv_yields_pristine_defaults(self, tmp_path: Path) -> None:
        repo, home = tmp_path / "untrusted-repo", tmp_path / "home"
        repo.mkdir()
        home.mkdir()
        (repo / ".env").write_text(_HOSTILE_DOTENV, encoding="utf-8")

        observed = _probe_config(cwd=repo, home=home)

        assert observed["provider"] == _field_default("provider")
        assert observed["local_code_model"] == _field_default("local_code_model")

    def test_explicit_config_file_is_still_honoured(self, tmp_path: Path) -> None:
        """Gate, do not delete: an operator naming a file in the real env still wins."""
        repo, home = tmp_path / "untrusted-repo", tmp_path / "home"
        repo.mkdir()
        home.mkdir()
        (repo / ".env").write_text(_HOSTILE_DOTENV, encoding="utf-8")
        operator_file = tmp_path / "operator.env"
        operator_file.write_text(
            f"TRELIX_EMBEDDER_LOCAL_CODE_MODEL={OPERATOR_MODEL}\n", encoding="utf-8"
        )

        observed = _probe_config(
            cwd=repo, home=home, extra_env={"TRELIX_CONFIG_FILE": str(operator_file)}
        )

        assert observed["local_code_model"] == OPERATOR_MODEL

    @pytest.mark.parametrize("anchor", ["home", "xdg"])
    def test_operator_config_directory_is_still_honoured(self, tmp_path: Path, anchor: str) -> None:
        repo, home = tmp_path / "untrusted-repo", tmp_path / "home"
        repo.mkdir()
        (repo / ".env").write_text(_HOSTILE_DOTENV, encoding="utf-8")
        config_home = home / ".config" if anchor == "home" else tmp_path / "xdg"
        (config_home / "trelix").mkdir(parents=True)
        (config_home / "trelix" / "env").write_text(
            f"TRELIX_EMBEDDER_LOCAL_CODE_MODEL={OPERATOR_MODEL}\n", encoding="utf-8"
        )
        extra = {} if anchor == "home" else {"XDG_CONFIG_HOME": str(config_home)}

        observed = _probe_config(cwd=repo, home=home, extra_env=extra)

        assert observed["local_code_model"] == OPERATOR_MODEL


class TestReviewWorkflowDoesNotTrustItsOwnCheckout:
    """The CI leg of the same defect — the only one that needs no human at all.

    ``.github/workflows/trelix-review.yml`` triggers on ``pull_request``, checks the
    PR head into the workspace and runs ``trelix index .`` with the cwd inside it.
    A fork PR gets a read-only token and no secrets, but a same-repo branch PR gets
    the ``pull-requests: write`` / ``checks: write`` the job declares.
    """

    @staticmethod
    def _steps() -> list[dict]:
        import yaml

        workflow = Path(__file__).parents[2] / ".github/workflows/trelix-review.yml"
        assert workflow.is_file(), f"missing workflow: {workflow}"
        return yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["trelix-review"][
            "steps"
        ]

    def test_dotenv_is_removed_before_trelix_runs(self) -> None:
        """Order is the assertion: a strip step after indexing would prove nothing."""
        steps = self._steps()
        runs = [str(step.get("run", "")) for step in steps]
        strips = [i for i, run in enumerate(runs) if "rm -f .env" in run]
        invocations = [
            i for i, run in enumerate(runs) if "trelix index" in run or "trelix review" in run
        ]

        assert strips, "no step removes the PR checkout's .env"
        assert invocations, "premise changed: no trelix invocation in the workflow"
        assert min(strips) < min(invocations)

    def test_no_trelix_step_hides_its_own_failure(self) -> None:
        """`continue-on-error: true` on a step that executes repo content is silent."""
        offenders = [
            step.get("name")
            for step in self._steps()
            if "trelix " in str(step.get("run", "")) and step.get("continue-on-error") is True
        ]

        assert not offenders, f"continue-on-error hides failures in: {offenders}"


class TestResolveOperatorEnvFile:
    """The resolver in isolation — the two ways it must not reintroduce the cwd."""

    def test_relative_override_is_made_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative TRELIX_CONFIG_FILE resolved lazily would follow the cwd again."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRELIX_CONFIG_FILE", "operator.env")

        resolved = resolve_operator_env_file()

        assert resolved is not None
        assert resolved.is_absolute()
        assert resolved == (tmp_path / "operator.env").resolve()

    def test_no_home_and_no_passwd_entry_yields_no_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The container case: an arbitrary UID with no HOME must not break the import.

        Path.home() raises RuntimeError there, and letting it propagate would make
        ``import trelix.core.config`` fail rather than fall back to "no dotenv".
        """
        monkeypatch.delenv("TRELIX_CONFIG_FILE", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setattr(
            Path, "home", classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("no home")))
        )

        assert resolve_operator_env_file() is None
