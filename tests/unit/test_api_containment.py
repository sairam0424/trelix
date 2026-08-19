"""REST-layer containment: the allow-list of repository roots (SEC-01/02/12).

The defect these tests exist for is NOT a broken comparison. Every containment
check in ``api/app.py`` was written correctly — ``Path.resolve()`` then
``Path.is_relative_to()``, never ``str.startswith()`` — and the comments around
them reason correctly about why a prefix match would wrongly accept a sibling
``<repo>-evil``. The defect is that the *root* those sound comparisons are
anchored to came out of the same request body being validated:

    repo_root = Path(body.repo_path).resolve()   # the attacker's own input
    if not path.is_relative_to(repo_root): ...   # therefore always satisfiable

With ``repo_path="/"`` every file on the host is "inside repo_path", so
``POST /parse`` was a filtered arbitrary-file read and every ``repo=`` route was
a cross-repo source read. ``TRELIX_API_AUTH_TOKEN`` does not close it: there is
one shared secret and no authorization layer, so a token holder gets the same
primitive.

So these tests are written from the attacker's position: the malicious input is
constructed and the response is asserted to be a refusal, with the planted bytes
asserted absent from the body. A test that only checks that legitimate use still
returns 200 proves nothing about containment.

The trusted root now comes from outside the request — the path ``trelix serve``
was pointed at (``create_app(served_root=...)``) or an explicit
``TRELIX_ALLOWED_REPO_ROOTS``. The pattern is the one already shipped at
``packages/trelix-mcp/src/trelix_mcp/server.py`` (``_confine_federation_config_path``),
which the REST layer never adopted.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from trelix.api.app import create_app

# A distinctive token planted in files/bodies the caller must never receive.
# Asserting on the token (not just on the status code) is what makes these
# tests refutations of a *read*, rather than of a status code that could be
# produced by an unrelated failure.
CANARY = "leaked_canary_9f3a"


def _plant_python_file(directory: Path) -> Path:
    """Write a .py file whose parsed *signature* carries the canary.

    ``.py`` (not ``.env``/``id_rsa``) because ``detect_language`` returns
    UNKNOWN for unmapped names and ``/parse`` then answers an empty 200 — a
    canary the parser never surfaces would make a passing test meaningless.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "creds.py"
    target.write_text(f"def {CANARY}():\n    return 1\n", encoding="utf-8")
    return target


def _mock_search_ctx() -> MagicMock:
    """A retrieval context whose single result body carries the canary.

    ``Retriever`` is patched rather than driven for real: a real retrieve()
    would construct an embedder and reach a paid endpoint. What is under test
    is whether the route refuses the unserved repo *before* retrieval, so a
    recorder that would happily hand back the planted bytes is the strongest
    available stand-in — it fails loudly if the refusal is missing.
    """
    result = MagicMock()
    result.file.rel_path = "creds.py"
    result.symbol.qualified_name = "load_creds"
    result.symbol.kind.value = "function"
    result.symbol.line_start = 1
    result.symbol.line_end = 3
    result.symbol.body = f"def load_creds(): return {CANARY!r}"
    result.file.language.value = "python"
    result.score = 0.9
    result.source = "vector"
    ctx = MagicMock()
    ctx.results = [result]
    return ctx


@pytest.fixture(params=["served_root", "allowed_roots_env"])
def confined_app(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ANN401
    """Build an app whose ONLY allowed root is the directory passed in.

    Parametrized over both configuration surfaces because they must be
    interchangeable: ``trelix serve <repo>`` supplies one root positionally,
    while the deliberate multi-repo/federation model needs the explicit
    ``TRELIX_ALLOWED_REPO_ROOTS``. A fix that wired up only the argument would
    leave every containerized deployment (which has no CLI argument to give)
    unconfined.
    """

    def _build(root: Path) -> Any:  # noqa: ANN401
        if request.param == "served_root":
            return create_app(served_root=root)
        monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(root))
        return create_app()

    return _build


def _index_recorder() -> MagicMock:
    """Stand-in for Indexer that records the config it was handed.

    Never runs a real index: a real one walks, parses, chunks and POSTs every
    chunk to the configured embedding provider under the operator's key — which
    is exactly the abuse ``POST /index`` enabled, and must not be reproduced by
    a test.
    """
    recorder = MagicMock()
    recorder.return_value.index.return_value = {
        "files_found": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "symbols_extracted": 0,
        "chunks_total": 0,
        "chunks_embedded": 0,
        "errors": 0,
        "elapsed_seconds": 0.0,
    }
    return recorder


class TestParseContainment:
    """SEC-01 — ``POST /parse`` was a filtered arbitrary-file read."""

    def test_repo_path_of_root_is_refused(self, tmp_path: Path, confined_app: Any) -> None:  # noqa: ANN401
        """``repo_path="/"`` made the whole host "inside repo_path"."""
        served = tmp_path / "served"
        served.mkdir()
        secret = _plant_python_file(tmp_path / "private")

        client = TestClient(confined_app(served))
        resp = client.post("/parse", json={"repo_path": "/", "file_path": str(secret)})

        assert resp.status_code == 403, resp.text
        assert CANARY not in resp.text

    def test_absolute_file_path_outside_the_allowed_root_is_refused(
        self, tmp_path: Path, confined_app: Any
    ) -> None:  # noqa: ANN401
        """A truthful ``repo_path`` plus an absolute ``file_path`` under another
        directory. The repo-relative check already answered 400 here; the point
        of asserting 403 is that the refusal now comes from the allow-list,
        which is the layer that holds when repo_path itself is the lie."""
        served = tmp_path / "served"
        served.mkdir()
        secret = _plant_python_file(tmp_path / "private")

        client = TestClient(confined_app(served))
        resp = client.post("/parse", json={"repo_path": str(served), "file_path": str(secret)})

        assert resp.status_code == 403, resp.text
        assert CANARY not in resp.text

    def test_dot_dot_traversal_out_of_an_allowed_repo_is_refused(
        self, tmp_path: Path, confined_app: Any
    ) -> None:  # noqa: ANN401
        """A relative ``file_path`` is joined to repo_root before resolve(), so
        ../ segments are caught by the repo-relative layer — but only once
        repo_root is itself trustworthy, which is what the allow-list buys."""
        served = tmp_path / "served"
        served.mkdir()
        _plant_python_file(tmp_path / "private")

        client = TestClient(confined_app(served))
        resp = client.post(
            "/parse", json={"repo_path": str(served), "file_path": "../private/creds.py"}
        )

        assert resp.status_code in (400, 403), resp.text
        assert CANARY not in resp.text

    def test_a_file_inside_the_served_root_still_parses(
        self, tmp_path: Path, confined_app: Any
    ) -> None:  # noqa: ANN401
        """Containment must gate, not remove the feature."""
        served = tmp_path / "served"
        target = _plant_python_file(served / "src")

        client = TestClient(confined_app(served))
        resp = client.post("/parse", json={"repo_path": str(served), "file_path": str(target)})

        assert resp.status_code == 200, resp.text
        assert [s["name"] for s in resp.json()["symbols"]] == [CANARY]


class TestCrossRepoContainment:
    """SEC-02 — every route took ``repo`` as a bare absolute server path."""

    @pytest.mark.parametrize(
        "intent_hint",
        [None, "symbol_lookup", "feature_flow", "not_a_real_intent"],
        ids=["no-hint", "symbol_lookup", "feature_flow", "invalid-hint"],
    )
    def test_search_on_an_unserved_repo_is_refused(
        self, tmp_path: Path, intent_hint: str | None, confined_app: Any
    ) -> None:  # noqa: ANN401
        """``intent_hint`` selects a different retrieval strategy; containment
        must not be reachable through any of them."""
        served = tmp_path / "served"
        served.mkdir()
        private = tmp_path / "private"
        private.mkdir()

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = _mock_search_ctx()
            client = TestClient(confined_app(served))
            url = f"/search?query=load_creds&repo={private}"
            if intent_hint is not None:
                url += f"&intent_hint={intent_hint}"
            resp = client.get(url)

        assert resp.status_code == 403, resp.text
        assert CANARY not in resp.text
        # Refused before any retrieval — not merely filtered afterwards.
        MockRetriever.assert_not_called()

    def test_ask_on_an_unserved_repo_is_refused_with_a_status_code(self, tmp_path: Path) -> None:
        """``/ask`` is the one route whose failures are otherwise invisible: it
        returns a StreamingResponse, so anything that goes wrong inside the
        generator arrives as ``data: [ERROR: ...]`` inside an HTTP **200**.
        Containment has to answer before the response is committed, or the
        refusal would be a 200 that status-code monitoring never sees."""
        served = tmp_path / "served"
        served.mkdir()
        private = tmp_path / "private"
        private.mkdir()

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = _mock_search_ctx()
            client = TestClient(create_app(served_root=served))
            resp = client.get(f"/ask?query=load_creds&repo={private}")

        assert resp.status_code == 403, resp.text
        assert CANARY not in resp.text
        MockRetriever.assert_not_called()

    def test_index_of_an_unserved_root_is_refused_and_spends_nothing(self, tmp_path: Path) -> None:
        """``POST /index`` had no containment check at all, and it is the only
        route that spends the operator's embedding budget."""
        served = tmp_path / "served"
        served.mkdir()
        recorder = _index_recorder()

        with patch("trelix.indexing.indexer.Indexer", recorder):
            client = TestClient(create_app(served_root=served))
            resp = client.post("/index", json={"repo_path": "/"})

        assert resp.status_code == 403, resp.text
        recorder.assert_not_called()

    def test_index_with_repo_path_missing_is_422_not_500(self, tmp_path: Path) -> None:
        """``body: dict[str, str]`` gave FastAPI no schema, so a missing field
        was a KeyError inside the route: an unauthenticated 500 with a logged
        traceback instead of a 422. A containment refusal must not mask it
        either — no path was supplied, so this is a schema error."""
        served = tmp_path / "served"
        served.mkdir()
        client = TestClient(create_app(served_root=served), raise_server_exceptions=False)

        resp = client.post("/index", json={})

        assert resp.status_code == 422, resp.text

    def test_stats_on_an_unserved_directory_is_refused_and_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        """A read route must not write. ``/stats`` used to leave
        ``.trelix/{index.db,-wal,-shm,.gitignore}`` in any directory named."""
        served = tmp_path / "served"
        served.mkdir()
        fresh = tmp_path / "fresh"
        fresh.mkdir()

        client = TestClient(create_app(served_root=served))
        resp = client.get(f"/stats?repo={fresh}")

        assert resp.status_code == 403, resp.text
        assert os.listdir(fresh) == [], "a refused read route created files on disk"

    def test_an_empty_repo_value_is_refused_not_silently_read_as_the_cwd(
        self, tmp_path: Path
    ) -> None:
        """``?repo=`` (present, empty) must be refused, not treated as absent.

        ``Path("")`` is ``Path(".")``, so an empty value silently resolves to the
        server's own working directory — which is a directory nobody
        allow-listed, and on the shipped entrypoint is wherever the operator ran
        ``trelix serve`` from. Measured before this case existed: ``GET /stats?repo=``
        answered 200 and left ``.trelix/{index.db,.gitignore}`` in the process cwd.

        Skipping falsy values is the natural way to write the check and it is
        wrong; only genuinely ABSENT fields may fall through, so the route's own
        schema can answer 422.
        """
        served = tmp_path / "served"
        served.mkdir()
        client = TestClient(create_app(served_root=served), raise_server_exceptions=False)

        assert client.get("/stats?repo=").status_code == 403
        assert client.post("/index", json={"repo_path": ""}).status_code == 403

    def test_a_duplicated_repo_parameter_cannot_smuggle_an_unserved_path(
        self, tmp_path: Path
    ) -> None:
        """The dependency and the route must read the SAME value.

        Both go through ``QueryParams.get()``, which returns the last
        occurrence, so ``?repo=<allowed>&repo=<unserved>`` is refused rather
        than checked against one value and executed against the other. This is
        the classic way a gate that re-parses its input becomes decorative.
        """
        served = tmp_path / "served"
        served.mkdir()
        private = tmp_path / "private"
        private.mkdir()
        client = TestClient(create_app(served_root=served), raise_server_exceptions=False)

        assert client.get(f"/stats?repo={served}&repo={private}").status_code == 403
        assert client.get(f"/stats?repo={private}&repo={served}").status_code != 403
        assert os.listdir(private) == []

    def test_a_symlink_out_of_an_allowed_root_is_refused(self, tmp_path: Path) -> None:
        """resolve() runs before the comparison, so a symlink planted inside an
        allowed repo cannot be used to reach outside it. Worth an explicit case:
        the repositories this API serves are the untrusted content, and a
        symlink is the cheapest thing to plant in one."""
        served = tmp_path / "served"
        served.mkdir()
        secret = _plant_python_file(tmp_path / "private")
        (served / "escape").symlink_to(tmp_path / "private")

        client = TestClient(create_app(served_root=served))

        resp = client.post(
            "/parse",
            json={"repo_path": str(served), "file_path": str(served / "escape" / secret.name)},
        )
        assert resp.status_code == 403, resp.text
        assert CANARY not in resp.text
        assert client.get(f"/stats?repo={served / 'escape'}").status_code == 403

    def test_stats_on_an_unreadable_system_directory_is_refused(self, tmp_path: Path) -> None:
        """``repo=/etc`` raised an unhandled PermissionError (500). Containment
        answers before the filesystem is touched."""
        served = tmp_path / "served"
        served.mkdir()
        client = TestClient(create_app(served_root=served), raise_server_exceptions=False)

        resp = client.get("/stats?repo=/etc")

        assert resp.status_code == 403, resp.text


class TestServedRootPlumbing:
    """``trelix serve <repo_path>`` is the argument every operator reads as the
    scope. It reached only the startup banner."""

    def test_create_app_takes_a_served_root(self) -> None:
        params = inspect.signature(create_app).parameters
        assert "served_root" in params, f"create_app{inspect.signature(create_app)}"
        # Still callable with no arguments: uvicorn --factory and the existing
        # test estate construct it that way.
        assert params["served_root"].default is None

    def test_serve_passes_its_repo_path_into_create_app(self, tmp_path: Path) -> None:
        """Without this, the allow-list is dead code on the shipped entrypoint."""
        import uvicorn

        from trelix.cli.main import serve

        recorded: dict[str, Any] = {}

        def _fake_create_app(*args: Any, **kwargs: Any) -> object:
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return object()

        with (
            patch("trelix.api.app.create_app", _fake_create_app),
            patch.object(uvicorn, "run", lambda *a, **k: None),
        ):
            serve(str(tmp_path), "127.0.0.1", 8765)

        passed = recorded["kwargs"].get(
            "served_root", recorded["args"][0] if recorded["args"] else None
        )
        assert str(passed) == str(tmp_path)


class TestEveryGatedRouteIsConfined:
    """The dependency has to be on every gated route, so the NEXT route added
    cannot forget it.

    The route set is read off the live app, never off the decorators. It is
    filtered to endpoints defined in ``trelix.api.app``: the app also carries
    ``/openapi.json``, ``/docs``, ``/docs/oauth2-redirect`` and ``/redoc``,
    whose endpoints belong to ``fastapi.applications``. An unfiltered
    "every non-/health route" assertion would demand containment on the docs
    UI — the count is 14 total / 13 non-/health, and only the filtered 9 are
    trelix's own gated routes.
    """

    @staticmethod
    def _own_routes(app: Any) -> list[Any]:
        return [
            r
            for r in app.routes
            if getattr(getattr(r, "endpoint", None), "__module__", None) == "trelix.api.app"
        ]

    @staticmethod
    def _dep_names(route: Any) -> list[str]:
        return [getattr(d.dependency, "__name__", repr(d.dependency)) for d in route.dependencies]

    def test_every_own_non_health_route_carries_the_containment_dependency(self) -> None:
        app = create_app()
        own = self._own_routes(app)
        table = "\n".join(
            f"  {sorted(r.methods)} {r.path} deps={self._dep_names(r)}"
            for r in sorted(own, key=lambda r: r.path)
        )
        print(f"\ntotal routes {len(app.routes)} | endpoint in trelix.api.app {len(own)}\n{table}")

        gated = [r for r in own if r.path != "/health"]
        assert len(gated) == 9, table

        for route in gated:
            names = self._dep_names(route)
            assert "confine_repo" in names, f"{route.path} is not confined\n{table}"
            assert "authenticate" in names, f"{route.path} is not authenticated\n{table}"
            # 401 must win over 403: a caller with no credential learns nothing
            # about which roots are served.
            assert names.index("authenticate") < names.index("confine_repo"), route.path

    def test_health_is_not_confined(self) -> None:
        """Liveness probes carry no repo and must reach it without a token."""
        app = create_app()
        health = next(r for r in self._own_routes(app) if r.path == "/health")
        assert self._dep_names(health) == []


class TestAllowedRootsConfiguration:
    """``TRELIX_ALLOWED_REPO_ROOTS`` — the deliberate multi-repo/federation
    model needs more than one root."""

    def test_no_root_configured_refuses_every_caller_supplied_path(self, tmp_path: Path) -> None:
        """Fail closed. An unconfigured app that accepted any path is the
        vacuous state this whole change exists to remove."""
        client = TestClient(create_app())
        resp = client.get(f"/stats?repo={tmp_path}")
        assert resp.status_code == 403, resp.text

    def test_multiple_roots_are_all_accepted(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        first = tmp_path / "one"
        second = tmp_path / "two"
        for root in (first, second):
            root.mkdir()
        monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", os.pathsep.join([str(first), str(second)]))

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = MagicMock(results=[])
            client = TestClient(create_app())
            for root in (first, second):
                assert client.get(f"/search?query=x&repo={root}").status_code == 200

    def test_a_sibling_sharing_the_root_prefix_is_refused(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """``<root>-evil`` starts with the same characters as an allowed root.
        The allow-list compares with is_relative_to(), not startswith() — the
        same property the per-route checks already had, now applied to the root
        itself."""
        allowed = tmp_path / "repo"
        allowed.mkdir()
        evil = tmp_path / "repo-evil"
        evil.mkdir()
        monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(allowed))

        client = TestClient(create_app())
        assert client.get(f"/stats?repo={evil}").status_code == 403

    def test_the_refusal_does_not_disclose_the_configured_roots(self, tmp_path: Path) -> None:
        """The 403 body is an unauthenticated response. Echoing the allow-list
        (as the MCP helper does, where the caller is the local operator) would
        hand out absolute server paths for free."""
        served = tmp_path / "served_root_name_9f3a"
        served.mkdir()
        client = TestClient(create_app(served_root=served))

        resp = client.get(f"/stats?repo={tmp_path / 'elsewhere'}")

        assert resp.status_code == 403
        assert "served_root_name_9f3a" not in resp.text
        assert str(tmp_path) not in resp.text
