"""`suite.json`: the committed definition of one evaluation suite, and the checks before a clone.

A suite is ONE repository at ONE pinned commit, the golden file of queries about it, and the
file of frozen query plans recorded for those queries. The directory holds `suite.json` beside
the two files it names:

    {
      "schema_version": 1,
      "name": "trelix-self",                         also the directory name of the clone
      "golden_version": "2026-10-r1",                a label, frozen with the golden content
      "repo": {"url": "https://github.com/owner/name.git", "sha": "<40 hex>", "license": "MIT"},
      "golden": {"path": "golden.jsonl", "sha256": "<64 hex>"},
      "plans": {"path": "plans.jsonl", "sha256": "<64 hex>"}
    }

Every key is required, an unknown or a repeated key is refused, and all the problems found are
reported together. The two sha256 values are over the RAW bytes of the files as committed, so a
changed byte, a line-ending rewrite or a missing trailing newline refuses the suite.

Nothing here touches git or the network: `load_suite` proves the files, `input_problems` proves
that they are usable together, and only then does `trelix.eval.suite_git` clone anything.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trelix.eval._loading import ProblemsError, clip, read_bytes_capped, read_text_capped, show
from trelix.eval.golden_validate import validate_golden
from trelix.eval.harness import _parse_golden
from trelix.eval.results import SUITE_PATTERNS
from trelix.retrieval.planner.agent import PlanCacheMissError, _FrozenPlanCache
from trelix.retrieval.planner.models import QueryPlan

SUITE_SCHEMA_VERSION = 1
MAX_SUITE_BYTES = 64 * 1024
MAX_DATA_FILE_BYTES = 32 * 1024 * 1024

# A committed suite.json can arrive in a pull request, and a CI job that runs `eval-suite` would
# then send a git request to whatever host it names (localhost, an internal address, a cloud
# metadata endpoint). So a `repo.url` is https on one of these hosts and nothing else; adding a
# host is this one line, and a reviewed change.
ALLOWED_REPO_HOSTS = ("github.com",)

TOP_KEYS = ("schema_version", "name", "golden_version", "repo", "golden", "plans")
REPO_KEYS = ("url", "sha", "license")
FILE_KEYS = ("path", "sha256")

# https://<host>/<owner>/<repo>[.git]. The host is everything up to the first `/`, so a userinfo,
# a port, a trailing dot or a lookalike suffix makes it differ from ALLOWED_REPO_HOSTS. The owner
# is GitHub's rule for a user or organisation (ASCII letters and digits, single hyphens inside,
# 39 characters at most); the repository is GitHub's (letters, digits, `-`, `_` and `.`, up to
# 100). No query, no fragment and no further path can match.
_REPO_URL_RE = re.compile(
    r"https://(?P<host>[^/]+)/"
    r"(?P<owner>(?=[A-Za-z0-9-]{1,39}/)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,100})"
)
# A local repository, for the tests only: see `load_suite(allow_local=True)`.
_LOCAL_URL_RE = re.compile(r"(?:file://)?/[^\x00-\x1f\x7f]*")
_LICENSE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}")
_FILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_UNCOVERED_LISTED = 5
_QUERY_CLIP_CHARS = 80


class SuiteError(ProblemsError):
    """A suite that cannot be used; `problems` lists every reason found."""


@dataclass(frozen=True)
class SuiteSpec:
    """A validated `suite.json` whose golden and plans files match their recorded hashes."""

    name: str
    golden_version: str
    repo_url: str
    repo_sha: str
    license: str
    golden_file: Path
    golden_sha256: str
    plans_file: Path
    plans_sha256: str


class _PlanRequested(Exception):
    """The frozen plan cache asked for a live draw: its file holds no plans."""


def _no_draw() -> QueryPlan:
    raise _PlanRequested


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    found: dict[str, object] = {}
    for key, value in pairs:
        if key in found:
            raise ValueError(f"duplicate key {show(key)}")
        found[key] = value
    return found


def _reject_constant(name: str) -> float:
    raise ValueError(f"the constant {name} is not allowed")


# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------


def _exact_keys(where: str, block: object, keys: Sequence[str]) -> list[str]:
    if not isinstance(block, dict):
        return [f"{where} must be an object with the keys {', '.join(keys)}"]
    unsupported = [f"{where}: unsupported key {show(key)}" for key in block if key not in keys]
    missing = [f"{where}: missing key {key!r}" for key in keys if key not in block]
    return unsupported + missing


def _pattern_problem(where: str, value: object, pattern: re.Pattern[str]) -> list[str]:
    if isinstance(value, str) and pattern.fullmatch(value):
        return []
    return [f"{where} must match {pattern.pattern} (got {show(value)})"]


def _is_allowed_repo_url(url: str) -> bool:
    match = _REPO_URL_RE.fullmatch(url)
    if match is None or match["host"] not in ALLOWED_REPO_HOSTS:
        return False
    # `.` and `..` would climb, and `.git` alone has no name before its suffix.
    return match["repo"].removesuffix(".git") not in ("", ".", "..")


def _url_problems(url: object, allow_local: bool) -> list[str]:
    if isinstance(url, str):
        if allow_local and _LOCAL_URL_RE.fullmatch(url):
            return []
        if _is_allowed_repo_url(url):
            return []
    return [
        "repo.url must be https://<host>/<owner>/<repo>[.git] with <host> one of "
        f"{', '.join(ALLOWED_REPO_HOSTS)}, and no credentials, port, query or fragment "
        f"(got {show(url)})"
    ]


def _repo_problems(repo: object, allow_local: bool) -> list[str]:
    problems = _exact_keys("repo", repo, REPO_KEYS)
    if not isinstance(repo, dict):
        return problems
    if "url" in repo:
        problems += _url_problems(repo["url"], allow_local)
    if "sha" in repo:
        problems += _pattern_problem(
            "repo.sha (a full commit id)", repo["sha"], SUITE_PATTERNS["repo_sha"]
        )
    if "license" in repo:
        problems += _pattern_problem(
            "repo.license (an SPDX identifier)", repo["license"], _LICENSE_RE
        )
    return problems


def _file_problems(where: str, block: object) -> list[str]:
    problems = _exact_keys(where, block, FILE_KEYS)
    if not isinstance(block, dict):
        return problems
    if "path" in block:
        problems += _pattern_problem(
            f"{where}.path (a bare file name)", block["path"], _FILE_NAME_RE
        )
    if "sha256" in block:
        problems += _pattern_problem(
            f"{where}.sha256", block["sha256"], SUITE_PATTERNS["golden_sha256"]
        )
    return problems


def _schema_problems(raw: object, allow_local: bool) -> list[str]:
    """Every way `raw` departs from the suite.json schema; `[]` when it is valid."""
    if not isinstance(raw, dict):
        return ["suite.json must be an object with the keys " + ", ".join(TOP_KEYS)]
    problems = _exact_keys("suite.json", raw, TOP_KEYS)
    version = raw.get("schema_version")
    if "schema_version" in raw and not (type(version) is int and version == SUITE_SCHEMA_VERSION):
        problems.append(
            f"unsupported schema_version {show(version)}: this reader understands "
            f"{SUITE_SCHEMA_VERSION}"
        )
    for key in ("name", "golden_version"):
        if key in raw:
            problems += _pattern_problem(key, raw[key], SUITE_PATTERNS[key])
    if "repo" in raw:
        problems += _repo_problems(raw["repo"], allow_local)
    for key in ("golden", "plans"):
        if key in raw:
            problems += _file_problems(key, raw[key])
    return problems


# ---------------------------------------------------------------------------
# the files and their hashes
# ---------------------------------------------------------------------------


def _verified_file(suite_dir: Path, what: str, block: Mapping[str, str]) -> Path:
    """The file `block` names, when it is a regular file of the suite directory with its hash.

    `resolve` runs first, so a name that is a symlink to a file elsewhere is refused however
    the link is spelled. Raises SuiteError naming both digests when the bytes differ.
    """
    name, expected = block["path"], block["sha256"]
    try:
        path = (suite_dir / name).resolve(strict=True)
    except (OSError, RuntimeError) as exc:  # RuntimeError: a symlink loop on Python 3.11, 3.12
        raise SuiteError(
            [f"{what}: cannot find {name!r} beside suite.json: {clip(str(exc))}"]
        ) from exc
    if path.parent != suite_dir or not path.is_file():
        raise SuiteError([f"{what}: {name!r} must be a regular file inside the suite directory"])
    try:
        data = read_bytes_capped(path, MAX_DATA_FILE_BYTES)
    except (OSError, ValueError) as exc:
        raise SuiteError([f"{what}: cannot read {name!r}: {clip(str(exc))}"]) from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise SuiteError([f"{what}: sha256 of {name} is {actual} but suite.json says {expected}"])
    return path


def load_suite(path: Path | str, *, allow_local: bool = False) -> SuiteSpec:
    """Read `suite.json`, validate it and verify the hashes of the files it names.

    Raises SuiteError listing every problem found. Both files are hashed before anything else
    is done with them, and the hash problems of the two are reported together.

    `repo.url` must name a host of `ALLOWED_REPO_HOSTS`. `allow_local` also accepts a local path
    or a `file://` URL, so that the tests can clone a repository built in a temp directory; the
    command line never sets it, so a `suite.json` cannot select a local or an internal remote.
    """
    suite_file = Path(path)
    try:
        text = read_text_capped(suite_file, MAX_SUITE_BYTES)
    except (OSError, ValueError) as exc:
        raise SuiteError([f"cannot read suite.json: {exc}"]) from exc
    try:
        raw = json.loads(
            text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError) as exc:
        raise SuiteError([f"suite.json is not valid JSON: {clip(str(exc), 200)}"]) from exc
    problems = _schema_problems(raw, allow_local)
    if problems:
        raise SuiteError(problems)

    suite_dir = suite_file.resolve().parent
    files: dict[str, Path] = {}
    for what in ("golden", "plans"):
        try:
            files[what] = _verified_file(suite_dir, what, raw[what])
        except SuiteError as exc:
            problems += exc.problems
    if problems:
        raise SuiteError(problems)
    return SuiteSpec(
        name=raw["name"],
        golden_version=raw["golden_version"],
        repo_url=raw["repo"]["url"],
        repo_sha=raw["repo"]["sha"],
        license=raw["repo"]["license"],
        golden_file=files["golden"],
        golden_sha256=raw["golden"]["sha256"],
        plans_file=files["plans"],
        plans_sha256=raw["plans"]["sha256"],
    )


# ---------------------------------------------------------------------------
# the checks that run before a clone
# ---------------------------------------------------------------------------


def input_problems(spec: SuiteSpec) -> list[str]:
    """What is wrong with the golden and plans files taken together; `[]` when they are usable.

    The golden file must validate as `trelix eval-validate` validates it, without its
    repository (the paths are checked after the clone) and without its v2 strata thresholds
    (those decide whether a golden file may be committed, not whether a suite may run).
    """
    report = validate_golden(spec.golden_file, tree=None, min_per_stratum=0, min_validated=0.0)
    if report.violations:
        return [f"golden: {violation}" for violation in report.violations]
    try:
        queries = [entry.query for entry in _parse_golden(spec.golden_file)]
    except ValueError as exc:
        return [f"golden: {clip(str(exc), 200)}"]
    return _plan_problems(spec.plans_file, queries)


def _plan_problems(plans_file: Path, queries: Sequence[str]) -> list[str]:
    """Why the plans file cannot be replayed for every one of `queries`, or `[]`.

    The file is read by the class the planner itself uses, asked for each query the way the
    planner asks (`project_context` is always None), so the key normalisation and the record
    rules cannot differ from the run. A file with no plans is refused, not treated as empty:
    the planner would read it as "record" and call the LLM for every query.
    """
    try:
        cache = _FrozenPlanCache(plans_file)
    except (ValueError, KeyError, TypeError) as exc:
        return [
            f"plans: {plans_file.name} is not usable: {type(exc).__name__}: {clip(str(exc), 200)}"
        ]
    uncovered: list[str] = []
    for query in queries:
        try:
            cache.plan(query, None, _no_draw)
        except PlanCacheMissError:
            uncovered.append(query)
        except _PlanRequested:
            return [
                f"plans: {plans_file.name} holds no plans. eval-suite never records plans: "
                "an empty plans file would make the planner call the LLM and record"
            ]
    if not uncovered:
        return []
    listed = ", ".join(clip(repr(q), _QUERY_CLIP_CHARS) for q in uncovered[:_UNCOVERED_LISTED])
    more = len(uncovered) - _UNCOVERED_LISTED
    tail = f", and {more} more" if more > 0 else ""
    return [
        f"plans: {len(uncovered)} of {len(queries)} golden queries have no recorded plan in "
        f"{plans_file.name}: {listed}{tail}"
    ]
