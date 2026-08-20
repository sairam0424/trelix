"""`packages` and `bin` are ignored only when they are not first-party source.

Those two names, plus `obj`, were added to the default `extra_ignore_dirs` for .NET build
output. They also silently exclude the source tree of every pnpm/npm/yarn/lerna monorepo
and every Node CLI's `bin/`. Measured in WALK-UNITS (files a bare `FileWalker.walk()`
yields, which is what trelix would index): 36 first-party files under Graph-Forge's
`packages/`, 168 under CommandVault's, 137 under ContextOS's, 104 under Tombstone's and 189
under MindForge's `bin/` — every one of them 0 indexed, while the run reported `errors: 0`.
CommandVault finished with 31 of 212 tracked files and exactly ONE file in a code language,
so its index reads as "this codebase has no call graph" rather than "the source was never
walked". See `_CONDITIONAL_IGNORE_DIRS` in walker.py for the full table.

Every walker test here asserts on the set of `rel_path`s `FileWalker.walk()` yields, never
on `_is_ignored_dir` directly: the classification rules are free to move as long as the
files that reach the index do not.

Two modes are exercised:
  * report-only (`index_conditional_dirs=False`, today's default) — the walk is unchanged
    and the reclassified directory is named at WARNING. This is what makes the release free.
  * enforcing (`index_conditional_dirs=True`) — the walk admits proven source. This is the
    code path that becomes the default once the walk-config fingerprint change ships with it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.indexing.walker import FileWalker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str = "export const x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, payload: object) -> Path:
    return _write(path, json.dumps(payload))


def _config(repo: Path, *, extra_ignore_dirs: list[str] | None = None) -> IndexConfig:
    """A config whose walker settings are trelix's shipped defaults unless overridden."""
    walker_kwargs: dict[str, object] = {}
    if extra_ignore_dirs is not None:
        walker_kwargs["extra_ignore_dirs"] = extra_ignore_dirs
    return IndexConfig(repo_path=str(repo), walker=WalkerConfig(**walker_kwargs))  # type: ignore[arg-type]


def _rel_paths(walker: FileWalker) -> set[str]:
    """The repo-relative paths a walk yields, as posix strings."""
    return {Path(f.rel_path).as_posix() for f in walker.walk()}


def _walk(repo: Path, *, extra_ignore_dirs: list[str] | None = None) -> set[str]:
    """Walk with the conditional tier ENFORCING — the Stage-2 default."""
    return _rel_paths(
        FileWalker(_config(repo, extra_ignore_dirs=extra_ignore_dirs), index_conditional_dirs=True)
    )


def _walk_today(repo: Path, *, extra_ignore_dirs: list[str] | None = None) -> set[str]:
    """Walk exactly as every shipped version does, probe in report-only mode."""
    return _rel_paths(FileWalker(_config(repo, extra_ignore_dirs=extra_ignore_dirs)))


# trelix's own shipped list, as a mutable copy the customisation tests can add to and
# subtract from. Read from `WalkerConfig` rather than restated, so it cannot drift.
_SHIPPED_IGNORE_DIRS = list(WalkerConfig().extra_ignore_dirs)


def _reported(caplog: pytest.LogCaptureFixture) -> set[str]:
    """The directories named by the report, keyed on the leading path in each message."""
    return {r.getMessage().split(" is NOT being indexed")[0] for r in caplog.records}


# ---------------------------------------------------------------------------
# Fixture shapes, each named after the real repository it was measured on
# ---------------------------------------------------------------------------


def _pnpm_workspace_no_key(repo: Path) -> None:
    """CommandVault shape: a marker file, and a `package.json` with NO `workspaces` key."""
    _write(repo / "pnpm-workspace.yaml", "packages:\n  - 'packages/*'\n")
    _json(repo / "package.json", {"name": "commandvault"})
    _write(repo / "packages/core/index.ts")
    _write(repo / "packages/cli/main.ts")


def _npm_workspaces_no_marker(repo: Path) -> None:
    """Graph-Forge / Tombstone shape: a `workspaces` key and NO marker file."""
    _json(repo / "package.json", {"name": "gf", "workspaces": ["packages/*"]})
    _write(repo / "packages/a/x.ts")


def _lerna(repo: Path) -> None:
    _json(repo / "lerna.json", {"packages": ["packages/*"]})
    _write(repo / "packages/a/x.ts")


def _node_cli_bin(repo: Path) -> None:
    """MindForge shape: `bin` declared in `package.json`, pointing into `bin/`."""
    _json(repo / "package.json", {"name": "mf", "bin": {"tool": "bin/cli.js"}})
    _write(repo / "bin/cli.js", "#!/usr/bin/env node\n")
    _write(repo / "bin/helper.js", "module.exports = 1\n")


def _workspace_and_cli(repo: Path) -> None:
    """Both conditional names live at once, from a single manifest.

    This is the shape that exposed the report's old gate: following the warning's own advice
    for `packages` used to silence the warning about `bin` too.
    """
    _json(
        repo / "package.json",
        {"name": "mf", "workspaces": ["packages/*"], "bin": {"mf": "bin/cli.js"}},
    )
    _write(repo / "packages/core/index.ts")
    _write(repo / "bin/cli.js", "#!/usr/bin/env node\n")


def _dotnet_restore_beside_a_spa_workspace(repo: Path, marker: str | None) -> None:
    """ASP.NET plus an npm-workspace SPA — the shape where the .NET veto is load-bearing.

    A root `package.json` with a `workspaces` key supplies positive evidence, so the veto is
    the ONLY thing holding the NuGet restore out. That makes it the fixture that can tell a
    marker name trelix recognises from one it merely looks like it recognises: with `marker`
    None the restore IS admitted, which is what stops these tests passing vacuously.
    """
    _write(repo / "src/App/Program.cs", "class P {}\n")
    _json(repo / "package.json", {"name": "app", "workspaces": ["spa/*"]})
    if marker is not None:
        _write(repo / marker, "<Project />\n")
    _write(repo / "packages/Vendor.Lib.1.0.0/README.md", "# vendored\n")
    _write(repo / "packages/Vendor.Lib.1.0.0/contentFiles/any/any/settings.json", "{}\n")


def _dotnet_solution(repo: Path) -> None:
    """The regression the three entries exist to prevent: NuGet restore + MSBuild output."""
    _write(repo / "App.sln", "Microsoft Visual Studio Solution File\n")
    _write(repo / "packages/Newtonsoft.Json.13.0.1/lib/net6.0/README.md", "# vendored\n")
    _write(repo / "src/App/App.csproj", "<Project />\n")
    _write(repo / "src/App/Program.cs", "class P {}\n")
    _write(repo / "src/App/bin/Debug/Generated.cs", "// generated\n")
    _write(repo / "src/App/obj/Debug/AssemblyInfo.cs", "// generated\n")


# ---------------------------------------------------------------------------
# `packages` — the two evidence branches, and what must NOT count as evidence
# ---------------------------------------------------------------------------


class TestPackagesIsIndexedWhenItIsAWorkspace:
    def test_marker_file_alone_is_enough(self, tmp_path: Path) -> None:
        """Fixture 1. CommandVault has no `workspaces` key, so the marker is the only signal."""
        _pnpm_workspace_no_key(tmp_path)
        found = _walk(tmp_path)
        assert "packages/core/index.ts" in found
        assert "packages/cli/main.ts" in found

    def test_workspaces_key_alone_is_enough(self, tmp_path: Path) -> None:
        """Fixture 2. Graph-Forge has no marker file, so the parsed key is the only signal.

        This and the test above must both exist, or either branch can be deleted in
        silence while half the measured population keeps working.
        """
        _npm_workspaces_no_marker(tmp_path)
        assert "packages/a/x.ts" in _walk(tmp_path)

    def test_lerna_marker(self, tmp_path: Path) -> None:
        """Fixture 3."""
        _lerna(tmp_path)
        assert "packages/a/x.ts" in _walk(tmp_path)

    def test_turbo_json_alone_is_not_enough(self, tmp_path: Path) -> None:
        """Fixture 4. `turbo.json` appears in non-workspace repos, so it proves nothing.

        Pins a deliberate weakness: a turbo-only repo keeps today's behaviour and is
        reported rather than silently re-included.
        """
        _json(tmp_path / "turbo.json", {"pipeline": {}})
        _json(tmp_path / "package.json", {"name": "t"})
        _write(tmp_path / "packages/a/x.ts")
        assert "packages/a/x.ts" not in _walk(tmp_path)

    def test_malformed_package_json_falls_back_to_the_marker(self, tmp_path: Path) -> None:
        """Fixture 12. A parse failure degrades to "no key evidence" and never fails a walk."""
        _write(tmp_path / "package.json", "{not json")
        _write(tmp_path / "pnpm-workspace.yaml", "packages:\n  - 'packages/*'\n")
        _write(tmp_path / "packages/a/x.ts")
        found = _walk(tmp_path)  # must not raise
        assert "packages/a/x.ts" in found

    def test_evidence_is_per_parent_not_inherited(self, tmp_path: Path) -> None:
        """Fixture 13. Workspace evidence at the root says nothing about a nested `packages/`."""
        _json(tmp_path / "package.json", {"name": "r", "workspaces": ["packages/*"]})
        _write(tmp_path / "packages/a/x.ts")
        _write(tmp_path / "a/b/c/d/packages/y.ts")
        found = _walk(tmp_path)
        assert "packages/a/x.ts" in found
        assert "a/b/c/d/packages/y.ts" not in found

    def test_never_reclassified_under_a_package_store(self, tmp_path: Path) -> None:
        """Fixture 14. A store holds copies of other projects, manifests included.

        `.pnpm-store` is NOT in `extra_ignore_dirs`, so the walk genuinely reaches inside
        it — this rule is the only thing keeping a CAS store's `packages/` out.
        """
        _write(tmp_path / "pnpm-workspace.yaml", "packages:\n  - 'packages/*'\n")
        store = tmp_path / ".pnpm-store/v10/projects/deadbeef"
        _write(store / "pnpm-workspace.yaml", "packages:\n  - 'packages/*'\n")
        _write(store / "packages/dep/index.js", "module.exports = 1\n")
        found = _walk(tmp_path)
        assert ".pnpm-store/v10/projects/deadbeef/packages/dep/index.js" not in found, found
        # The store's own manifest is yielded, as it is on every shipped version —
        # `.pnpm-store` is not in `extra_ignore_dirs` and this change does not add it.
        assert ".pnpm-store/v10/projects/deadbeef/pnpm-workspace.yaml" in found


# ---------------------------------------------------------------------------
# `bin` — positive evidence required, because absence of .NET is not evidence of source
# ---------------------------------------------------------------------------


class TestBinNeedsADeclaration:
    def test_declared_bin_object_form(self, tmp_path: Path) -> None:
        """Fixture 7. The only shape that admits a `bin/`."""
        _node_cli_bin(tmp_path)
        found = _walk(tmp_path)
        assert "bin/cli.js" in found
        assert "bin/helper.js" in found

    def test_declared_bin_bare_string_form(self, tmp_path: Path) -> None:
        """Fixture 10. npm allows `"bin": "bin/cli.js"`."""
        _json(tmp_path / "package.json", {"name": "x", "bin": "bin/cli.js"})
        _write(tmp_path / "bin/cli.js", "#!/usr/bin/env node\n")
        assert "bin/cli.js" in _walk(tmp_path)

    def test_package_json_without_a_bin_field(self, tmp_path: Path) -> None:
        """Fixture 8. Ambiguous stays ignored — cheap and safe, and it is reported."""
        _json(tmp_path / "package.json", {"name": "x"})
        _write(tmp_path / "bin/cli.js", "#!/usr/bin/env node\n")
        assert "bin/cli.js" not in _walk(tmp_path)

    def test_declaration_pointing_elsewhere(self, tmp_path: Path) -> None:
        """Fixture 11. `{"bin": {"tool": "dist/cli.js"}}` says nothing about `bin/`."""
        _json(tmp_path / "package.json", {"name": "x", "bin": {"tool": "dist/cli.js"}})
        _write(tmp_path / "bin/x.js", "module.exports = 1\n")
        assert "bin/x.js" not in _walk(tmp_path)

    @pytest.mark.parametrize(
        ("declared", "admitted"),
        [
            ("./bin/cli.js", True),
            ("bin/cli.js", True),
            ("../bin/cli.js", False),
            ("/usr/local/bin/cli.js", False),
            ("bin", False),
        ],
    )
    def test_only_a_target_inside_the_directory_admits_it(
        self, tmp_path: Path, declared: str, admitted: bool
    ) -> None:
        """A leading `./` is a relative-path idiom; `../` and an absolute path are not.

        `"bin": "bin"` names a file, not a directory, and must not admit the directory
        that shares its name.
        """
        _json(tmp_path / "package.json", {"name": "x", "bin": declared})
        _write(tmp_path / "bin/cli.js", "#!/usr/bin/env node\n")
        assert ("bin/cli.js" in _walk(tmp_path)) is admitted

    def test_virtualenv_bin_needs_no_special_rule(self, tmp_path: Path) -> None:
        """Fixture 9. Requiring positive evidence subsumes a virtualenv-specific rule."""
        _write(tmp_path / ".venv-proto/pyvenv.cfg", "home = /usr\n")
        _write(tmp_path / ".venv-proto/bin/activate.sh", "echo hi\n")
        _write(tmp_path / ".venv-proto/bin/tool.py", "x = 1\n")
        assert _walk(tmp_path) == set()


# ---------------------------------------------------------------------------
# The .NET regression the entries exist to prevent
# ---------------------------------------------------------------------------


class TestDotNetOutputStaysExcluded:
    def test_solution_keeps_packages_bin_and_obj_out(self, tmp_path: Path) -> None:
        """Fixture 5. Restored NuGet packages and MSBuild output, first-party source kept."""
        _dotnet_solution(tmp_path)
        found = _walk(tmp_path)
        assert found == {"src/App/Program.cs", "src/App/App.csproj"}, found

    def test_dotnet_wins_a_tie(self, tmp_path: Path) -> None:
        """Fixture 6. Both signals present -> stays ignored. Cheap-safe direction."""
        _write(tmp_path / "App.sln", "Microsoft Visual Studio Solution File\n")
        _write(tmp_path / "pnpm-workspace.yaml", "packages:\n  - 'packages/*'\n")
        _write(tmp_path / "packages/x/a.ts")
        assert "packages/x/a.ts" not in _walk(tmp_path)

    def test_obj_stays_unconditional(self, tmp_path: Path) -> None:
        """Fixture 16. Dropping `obj` re-admitted 0 files in all six repos; not conditional.

        That 0 is "untested" rather than "harmless": none of the six HAS a reachable .NET
        `obj/` — the workspace's only two are a gitignored Go toolchain cache and one inside
        `node_modules`. This fixture is therefore the only place the claim is exercised at all.
        """
        _write(tmp_path / "pnpm-workspace.yaml", "packages:\n  - 'packages/*'\n")
        _write(tmp_path / "obj/Generated.cs", "// generated\n")
        assert "obj/Generated.cs" not in _walk(tmp_path)

    @pytest.mark.parametrize(
        "marker",
        [
            # Control: the one casing that already worked, so a green row below means the
            # case-fold fired rather than the whole veto having been widened to everything.
            "Directory.Build.props",
            # Casings MSBuild and NuGet resolve identically, on the two case-insensitive
            # filesystems .NET is actually developed on. Each of these measured warn=1
            # junk=+50 against a 25-package restore before the comparison was case-folded.
            "Directory.build.props",
            "directory.build.props",
            "DIRECTORY.BUILD.PROPS",
            "Directory.Packages.props",
            "directory.packages.props",
            "packages.config",
            "Packages.config",
            # Names that were not markers at any casing. `NuGet.config` is the worst of
            # them: it declares `repositoryPath` and therefore MAKES a root `packages/` a
            # restore directory.
            "NuGet.config",
            "nuget.config",
            "packages.lock.json",
            "Directory.Build.targets",
            # Suffixes. `.sln`/`.csproj` were already case-folded and are controls; `.slnf`
            # (solution filter) and `.vcxproj` (C++ MSBuild) were missing entirely.
            "App.sln",
            "App.SLN",
            "App.slnf",
            "App.vcxproj",
            "Lib.csproj",
        ],
    )
    def test_every_marker_casing_keeps_a_nuget_restore_out(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, marker: str
    ) -> None:
        """A .NET marker must veto at every casing the toolchain accepts, and stay quiet.

        Table-driven so adding a marker cannot regress one casing of another: a new entry is
        one row, and a row that stops vetoing is one failure naming the exact filename.
        """
        _dotnet_restore_beside_a_spa_workspace(tmp_path, marker)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            today = _walk_today(tmp_path)

        assert not any(p.startswith("packages/") for p in _walk(tmp_path)), marker
        assert "src/App/Program.cs" in today
        assert caplog.records == [], [r.getMessage() for r in caplog.records]

    def test_the_same_tree_without_a_marker_is_admitted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The control for the table above: strip the marker and the restore comes back.

        Without this the table could pass because the veto fires unconditionally — which
        would silently re-hide every genuine monorepo.
        """
        _dotnet_restore_beside_a_spa_workspace(tmp_path, None)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            _walk_today(tmp_path)

        assert "packages/Vendor.Lib.1.0.0/README.md" in _walk(tmp_path)
        assert _reported(caplog) == {"packages"}

    def test_unconditional_tier_survives_a_gitignore_negation(self, tmp_path: Path) -> None:
        """Fixture 15. Guards against routing unconditional names through the new path.

        `node_modules` is excluded by NAME before `.gitignore` is consulted, so a
        re-including negation must not resurrect it.
        """
        _write(tmp_path / ".gitignore", "!node_modules/\n!node_modules/**\n")
        _write(tmp_path / "node_modules/left-pad/index.js", "module.exports = 1\n")
        assert "node_modules/left-pad/index.js" not in _walk(tmp_path)


# ---------------------------------------------------------------------------
# The conditional tier may only WIDEN the unconditional tier, never narrow it
# ---------------------------------------------------------------------------


class TestCapitalisedConditionalDirsAreNeverDropped:
    """`Bin/` and `Packages/` are not in `extra_ignore_dirs` — nothing may exclude them here.

    The unconditional tier is byte-exact by design, so a capitalised directory is one the
    walk has always descended into. The conditional tier matches case-INSENSITIVELY, which is
    correct (`Packages/` and `packages/` are the same directory on a case-insensitive
    filesystem, and the probe answers for the real one) but means these directories enter the
    conditional block without ever having been excluded. Enforcing mode used to answer them
    with a bare `return True`, which DROPPED them — and silently, because the report is
    guarded on the same byte-exact match. Measured on the fixtures below: 7 files -> 1 for
    `Bin/`, 8 -> 1 for `Packages/`, 0 warnings either way.
    """

    @pytest.mark.parametrize("dir_name", ["Bin", "BIN", "Packages", "PACKAGES"])
    def test_source_survives_both_modes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, dir_name: str
    ) -> None:
        """No evidence, no exclusion: the walk must be identical in both modes.

        Deliberately given NO workspace manifest and no `bin` declaration, because that is
        the case the old code got wrong — evidence would have admitted the directory anyway
        and hidden the defect.
        """
        _write(tmp_path / "README.md", "# r\n")
        for i in range(6):
            _write(tmp_path / f"{dir_name}/deploy{i}.sh", "echo hi\n")
        expected = {"README.md", *(f"{dir_name}/deploy{i}.sh" for i in range(6))}

        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            assert _walk_today(tmp_path) == expected
            assert _walk(tmp_path) == expected
        assert caplog.records == [], [r.getMessage() for r in caplog.records]

    @pytest.mark.parametrize("dir_name", ["Bin", "Packages"])
    def test_a_gitignore_still_excludes_them(self, tmp_path: Path, dir_name: str) -> None:
        """Widening must not overshoot into "always index": `.gitignore` still decides.

        The replacement verdict defers to `_is_gitignored` rather than returning False, so a
        capitalised directory the repository itself ignores stays out in both modes.
        """
        _write(tmp_path / ".gitignore", f"{dir_name}/\n")
        _write(tmp_path / "README.md", "# r\n")
        _write(tmp_path / f"{dir_name}/deploy.sh", "echo hi\n")
        assert _walk_today(tmp_path) == {"README.md"}
        assert _walk(tmp_path) == {"README.md"}

    def test_the_lowercase_name_is_still_excluded_in_both_modes(self, tmp_path: Path) -> None:
        """The control. Widening the capitalised case must not un-exclude the real entry."""
        _write(tmp_path / "README.md", "# r\n")
        _write(tmp_path / "bin/deploy.sh", "echo hi\n")
        _write(tmp_path / "packages/x/a.ts")
        assert _walk_today(tmp_path) == {"README.md"}
        assert _walk(tmp_path) == {"README.md"}


# ---------------------------------------------------------------------------
# Report-only mode: this release must cost nothing
# ---------------------------------------------------------------------------


class TestReportOnlyModeIsFree:
    @pytest.mark.parametrize(
        ("build", "reclassified", "evidence"),
        [
            (_pnpm_workspace_no_key, "packages", "pnpm-workspace.yaml"),
            (_npm_workspaces_no_marker, "packages", "package.json"),
            (_lerna, "packages", "lerna.json"),
            (_node_cli_bin, "bin", "package.json"),
        ],
    )
    def test_walk_is_unchanged_and_the_directory_is_named(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        build: object,
        reclassified: str,
        evidence: str,
    ) -> None:
        """Fixture 18. Nothing under the directory is yielded, and a warning says so."""
        build(tmp_path)  # type: ignore[operator]
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            found = _walk_today(tmp_path)

        assert not any(p.startswith(f"{reclassified}/") for p in found), found
        messages = [r.getMessage() for r in caplog.records]
        assert any(reclassified in m and evidence in m for m in messages), messages

    def test_a_customised_ignore_list_that_still_excludes_it_is_still_told(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fixture 19. Customising the list is a choice; still excluding `packages` is not.

        This used to assert silence, on the rationale that a hand-written override is a
        choice. The rationale is right and the gate was the wrong way to honour it: a list
        that STILL LISTS `packages` still hides `packages/core/index.ts`, so suppressing the
        report there just restores the silence the whole change exists to remove. Honouring
        the choice is `test_removing_the_name_is_the_only_way_to_silence_it` below.
        """
        _pnpm_workspace_no_key(tmp_path)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            found = _walk_today(tmp_path, extra_ignore_dirs=[".git", "node_modules", "packages"])

        assert not any(p.startswith("packages/") for p in found), found
        assert _reported(caplog) == {"packages"}

    def test_the_directory_is_named_once_not_once_per_verdict(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A watcher asks `_is_ignored_dir` once per event; the warning must not repeat."""
        _pnpm_workspace_no_key(tmp_path)
        walker = FileWalker(_config(tmp_path))
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            for _ in range(3):
                assert walker._is_ignored_dir(tmp_path / "packages") is True

        assert len(caplog.records) == 1, [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# The report must reach the users most likely to have customised the list
# ---------------------------------------------------------------------------


class TestTheReportFollowsTheEffectiveList:
    """Reported per NAME STILL LISTED, not per "the list is byte-identical to the default".

    docs/CONFIGURATION.md says the env var replaces the list, so "to add one entry you must
    restate the whole list". Doing exactly that used to silence the fix while changing nothing
    about what was hidden.
    """

    def test_adding_one_entry_does_not_silence_the_report(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The documented way to add an entry. `packages/` is still hidden, so still reported."""
        _pnpm_workspace_no_key(tmp_path)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            found = _walk_today(tmp_path, extra_ignore_dirs=[*_SHIPPED_IGNORE_DIRS, ".cache"])

        assert "packages/core/index.ts" not in found
        assert _reported(caplog) == {"packages"}

    def test_removing_one_name_still_reports_the_other(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The tail-eating case: acting on the advice for `packages` must not hide `bin/`."""
        _workspace_and_cli(tmp_path)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            found = _walk_today(
                tmp_path, extra_ignore_dirs=[d for d in _SHIPPED_IGNORE_DIRS if d != "packages"]
            )

        assert "packages/core/index.ts" in found, found
        assert "bin/cli.js" not in found, found
        assert _reported(caplog) == {"bin"}

    def test_removing_the_name_is_the_only_way_to_silence_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """How the choice is honoured. Both names dropped -> both indexed, nothing reported."""
        _workspace_and_cli(tmp_path)
        keep = [d for d in _SHIPPED_IGNORE_DIRS if d not in {"packages", "bin"}]
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            found = _walk_today(tmp_path, extra_ignore_dirs=keep)

        assert {"packages/core/index.ts", "bin/cli.js"} <= found, found
        assert caplog.records == [], [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# The watchers must agree with the walk, file for file
# ---------------------------------------------------------------------------


def _agreement_tree(repo: Path) -> None:
    _pnpm_workspace_no_key(repo)
    _write(repo / "obj/Generated.cs", "// generated\n")


def _file_watcher_verdict(walker: FileWalker, path: Path) -> bool:
    from trelix.indexing.watcher import FileWatcher

    watcher = FileWatcher.__new__(FileWatcher)
    watcher._walker = walker
    return watcher._should_index(str(path))


def _multi_watcher_verdict(walker: FileWalker, path: Path) -> bool:
    from trelix.indexing.multi_watcher import MultiRepoWatcher

    watcher = MultiRepoWatcher.__new__(MultiRepoWatcher)
    return watcher._should_index(walker, str(path))


class TestWatchersAgreeWithTheWalk:
    @pytest.mark.parametrize(
        "verdict", [_file_watcher_verdict, _multi_watcher_verdict], ids=["watch", "watch-all"]
    )
    def test_every_file_in_the_tree(self, tmp_path: Path, verdict: object) -> None:
        """Fixture 17. A row a watcher writes but a walk cannot reach is a phantom deletion.

        `compute_drift` reports it as `missing` and `--prune` offers to delete it, so the
        two surfaces disagreeing is not cosmetic. `FileWatcher._should_index` never applied
        `extra_ignore_dirs` at all: it returned True for `packages/core/index.ts` and
        `obj/Generated.cs` where the walk yields neither.
        """
        _agreement_tree(tmp_path)
        walker = FileWalker(_config(tmp_path))
        walked = {Path(f.rel_path).as_posix() for f in walker.walk()}

        for path in sorted(p for p in tmp_path.rglob("*") if p.is_file()):
            rel = path.relative_to(tmp_path).as_posix()
            assert verdict(walker, path) is (rel in walked), (  # type: ignore[operator]
                f"{rel}: watcher said {verdict(walker, path)}, walk said {rel in walked}"  # type: ignore[operator]
            )


# ---------------------------------------------------------------------------
# Both tiers are pinned, so a green suite means something to a reviewer
# ---------------------------------------------------------------------------


class TestTiersArePinned:
    def test_conditional_tier_membership(self) -> None:
        from trelix.indexing.walker import _CONDITIONAL_IGNORE_DIRS

        assert _CONDITIONAL_IGNORE_DIRS == {"packages", "bin"}
        assert "obj" not in _CONDITIONAL_IGNORE_DIRS

    def test_dotnet_marker_files_are_stored_lowercased(self) -> None:
        """The comparison case-folds one side, so the table must be pre-folded on the other.

        An entry added in its natural casing (`"Directory.Build.props"`) would silently never
        match `name.lower()`, which is exactly the bug this pins: a dead veto looks like a
        present one until a real solution walks through it.
        """
        from trelix.indexing.walker import _DOTNET_MARKER_FILES, _DOTNET_MARKER_SUFFIXES

        assert all(name == name.lower() for name in _DOTNET_MARKER_FILES), _DOTNET_MARKER_FILES
        assert all(s == s.lower() for s in _DOTNET_MARKER_SUFFIXES), _DOTNET_MARKER_SUFFIXES
        # `NuGet.config` declares `repositoryPath` and therefore MAKES a root `packages/` a
        # restore directory — the most direct proof of the shape these entries exist for.
        assert "nuget.config" in _DOTNET_MARKER_FILES

    def test_default_ignore_list_membership(self) -> None:
        """The exact default. Nothing else asserted more than `".trelix" in the list`."""
        assert set(WalkerConfig().extra_ignore_dirs) == {
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            "venv",
            ".venv",
            "env",
            "dist",
            "build",
            "target",
            "out",
            ".next",
            ".nuxt",
            "coverage",
            ".coverage",
            "vendor",
            "Pods",
            ".gradle",
            ".idea",
            ".vscode",
            ".angular",
            "bin",
            "obj",
            "packages",
            ".vs",
            ".rider",
            ".trelix",
        }
