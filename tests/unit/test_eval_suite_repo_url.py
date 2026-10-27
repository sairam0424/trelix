"""`repo.url`: `https://github.com/<owner>/<repo>[.git]` and nothing else.

A committed suite.json can arrive in a pull request and a CI job runs it, so a url naming any
other host (localhost, an internal address, the cloud metadata endpoint, a lookalike of github.com)
would be a server-side request from the runner. The host list is ONE constant,
`ALLOWED_REPO_HOSTS`, so adding a host is a one-line reviewed change. The rest of the schema is in
`test_eval_suite_spec.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `ALLOWED_REPO_HOSTS` widened                                   (another public host)
2. the host compared by prefix, by suffix, case-insensitively, or without its port or userinfo
                                                   (the lookalike, case, port and user cases)
3. a part of the url pattern dropped or loosened: the owner's hyphen rule or 39-character cap,
   the repo's 100-character cap, the `.` / `..` / `.git` refusal, the full match (a query,
   fragment, trailing slash, further path or trailing newline)
4. `allow_local` opening anything but a local path, or the refusal text not naming the hosts
                                 (test_allow_local_opens_..., test_a_host_is_added_in_one_place...)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.eval_suite_harness import write_suite
from trelix.eval.suite import SuiteError, load_suite

_SHA = "5fa2a032147f64d698077e8849e603872d76b5d0"


def _problems(path: Path, *, allow_local: bool = False) -> str:
    with pytest.raises(SuiteError) as caught:
        load_suite(path, allow_local=allow_local)
    return " | ".join(caught.value.problems)


_ACCEPTED_URLS = [
    ("with .git", "https://github.com/example/demo.git"),
    ("without .git", "https://github.com/example/demo"),
    ("one-letter names", "https://github.com/a/b"),
    (
        "hyphen in the owner, dot underscore hyphen in the repo",
        "https://github.com/Some-Org/r.n_x-1",
    ),
    ("a repo that starts with a dot", "https://github.com/a1-b2/.github"),
    ("an owner of 39 characters", "https://github.com/" + "a" * 39 + "/r"),
    ("a repo of 100 characters", "https://github.com/o/" + "r" * 100),
]

_REFUSED_URLS = [
    ("http", "http://github.com/o/r"),
    ("ssh", "ssh://git@github.com/o/r"),
    ("git protocol", "git://github.com/o/r"),
    ("scp-like", "git@github.com:o/r.git"),
    ("a user name", "https://user@github.com/o/r"),
    ("a user name and a password", "https://u:p@github.com/o/r"),
    ("github.com as the user of another host", "https://github.com@evil.example/o/r"),
    ("another host as the user of github.com", "https://evil.example@github.com/o/r"),
    ("a backslash before the @", "https://github.com\\@evil.example/o/r"),
    ("port 443", "https://github.com:443/o/r"),
    ("another port", "https://github.com:8443/o/r"),
    ("an IPv4 loopback literal", "https://127.0.0.1/o/r"),
    ("an IPv6 loopback literal", "https://[::1]/o/r"),
    ("a private address literal", "https://10.0.0.5/o/r"),
    ("the cloud metadata address", "https://169.254.169.254/o/r"),
    ("localhost", "https://localhost/o/r"),
    ("another public host", "https://gitlab.com/o/r"),
    ("a lookalike suffix", "https://github.com.evil.example/o/r"),
    ("a lookalike prefix", "https://evilgithub.com/o/r"),
    ("a subdomain", "https://www.github.com/o/r"),
    ("a trailing dot on the host", "https://github.com./o/r"),
    ("an upper-case host", "https://GITHUB.COM/o/r"),
    ("a mixed-case host", "https://GitHub.com/o/r"),
    ("a non-ASCII lookalike letter in the host", "https://g\u0456thub.com/o/r"),
    ("an owner of ..", "https://github.com/../r"),
    ("a repo of ..", "https://github.com/o/.."),
    ("a repo of .", "https://github.com/o/."),
    ("a repo of .git alone", "https://github.com/o/.git"),
    ("a repo of ..git", "https://github.com/o/..git"),
    ("percent-encoded dots in the owner", "https://github.com/%2e%2e/r"),
    ("a query", "https://github.com/o/r?x=1"),
    ("a fragment", "https://github.com/o/r#frag"),
    ("a trailing slash", "https://github.com/o/r/"),
    ("a further path", "https://github.com/o/r/tree/main"),
    ("an owner and no repo", "https://github.com/o"),
    ("no path", "https://github.com"),
    ("an empty owner", "https://github.com//r"),
    ("an owner that starts with a hyphen", "https://github.com/-o/r"),
    ("an owner that ends with a hyphen", "https://github.com/o-/r"),
    ("an owner with two hyphens in a row", "https://github.com/o--p/r"),
    ("an owner of 40 characters", "https://github.com/" + "a" * 40 + "/r"),
    ("a repo of 101 characters", "https://github.com/o/" + "r" * 101),
    ("an owner with an underscore", "https://github.com/o_x/r"),
    ("a non-ASCII owner", "https://github.com/\u00f6/r"),
    ("a trailing newline", "https://github.com/o/r\n"),
    ("a trailing space", "https://github.com/o/r "),
    ("an option", "--upload-pack=touch x"),
    ("the ext transport", "ext::sh -c id"),
    ("a local path", "/srv/demo.git"),
    ("a file url", "file:///srv/demo.git"),
    ("an empty string", ""),
]

_URL_PROBLEM = (
    "repo.url must be https://<host>/<owner>/<repo>[.git] with <host> one of github.com, "
    "and no credentials, port, query or fragment (got "
)


class TestRepoUrl:
    """`repo.url` is `https://github.com/<owner>/<repo>[.git]` and nothing else.

    A suite.json can arrive in a pull request and a CI job runs it, so any other host is a
    server-side request from the runner.
    """

    @pytest.mark.parametrize(("label", "url"), _ACCEPTED_URLS, ids=[u[0] for u in _ACCEPTED_URLS])
    def test_a_github_url_loads(self, tmp_path: Path, label: str, url: str) -> None:
        assert load_suite(write_suite(tmp_path / "suite", url, _SHA)).repo_url == url

    @pytest.mark.parametrize(("label", "url"), _REFUSED_URLS, ids=[u[0] for u in _REFUSED_URLS])
    def test_every_other_shape_is_refused_naming_the_allowed_host(
        self, tmp_path: Path, label: str, url: str
    ) -> None:
        with pytest.raises(SuiteError) as caught:
            load_suite(write_suite(tmp_path / "suite", url, _SHA))
        [problem] = caught.value.problems
        assert problem.startswith(_URL_PROBLEM)

    @pytest.mark.parametrize(
        "url",
        [
            "http://github.com/o/r",
            "https://127.0.0.1/o/r",
            "https://localhost/o/r",
            "https://evil.example/o/r",
            "https://github.com.evil.example/o/r",
        ],
    )
    def test_allow_local_opens_a_local_path_and_nothing_else(
        self, tmp_path: Path, url: str
    ) -> None:
        path = write_suite(tmp_path / "suite", url, _SHA)
        assert _problems(path, allow_local=True).startswith(_URL_PROBLEM)

    def test_a_host_is_added_in_one_place_and_the_message_lists_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_suite(tmp_path / "suite", "https://gitlab.com/o/r.git", _SHA)
        assert _problems(path).startswith(_URL_PROBLEM)
        monkeypatch.setattr("trelix.eval.suite.ALLOWED_REPO_HOSTS", ("github.com", "gitlab.com"))
        assert load_suite(path).repo_url == "https://gitlab.com/o/r.git"
        bad = write_suite(tmp_path / "other", "https://bitbucket.org/o/r", _SHA)
        assert "with <host> one of github.com, gitlab.com, and no credentials" in _problems(bad)
