"""Tests for the MCP resource subscription registry and the capability trelix-mcp reports.

Per MCP spec 2024-11-05 §Resources, ``capabilities.resources.subscribe=True`` tells a
client it may send ``resources/subscribe``. trelix-mcp registers no such handler (its
``subscribe_resource`` / ``unsubscribe_resource`` are plain tools), so it must report
``subscribe`` as false and must not force it to true.
"""

from __future__ import annotations

import pytest


class TestSubscriptionRegistry:
    def test_subscribe_stores_subscription(self):
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        reg.subscribe("trelix://repo//path/manifest", "sub-001")
        assert "sub-001" in reg.get_subscription_ids("trelix://repo//path/manifest")

    def test_unsubscribe_removes_subscription(self):
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        reg.subscribe("trelix://repo//path/manifest", "sub-001")
        reg.unsubscribe("sub-001")
        assert reg.get_subscription_ids("trelix://repo//path/manifest") == []

    def test_multiple_subscribers_same_uri(self):
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        reg.subscribe("trelix://repo//path/manifest", "sub-001")
        reg.subscribe("trelix://repo//path/manifest", "sub-002")
        ids = reg.get_subscription_ids("trelix://repo//path/manifest")
        assert "sub-001" in ids
        assert "sub-002" in ids

    def test_get_subscriptions_unknown_uri_returns_empty(self):
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        assert reg.get_subscription_ids("trelix://repo//unknown/manifest") == []

    def test_uri_for_subscription_id(self):
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        reg.subscribe("trelix://repo//path/manifest", "sub-001")
        assert reg.get_uri("sub-001") == "trelix://repo//path/manifest"

    def test_uri_unknown_subscription_returns_none(self):
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        assert reg.get_uri("sub-999") is None


class TestFileChangeNotificationBridge:
    def test_notify_subscribers_on_file_change(self):
        """When a watched file changes, notifications/resources/updated is sent
        to all subscribers."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.subscriptions import SubscriptionRegistry, notify_file_changed

        reg = SubscriptionRegistry()
        reg.subscribe("trelix://repo//path/manifest", "sub-001")
        reg.subscribe("trelix://repo//path/manifest", "sub-002")

        mock_send = MagicMock()
        with patch("trelix_mcp.subscriptions.send_resource_notification", mock_send):
            notify_file_changed(
                registry=reg,
                repo_path="/path",
                changed_file="src/auth.py",
            )

        # Both subscribers should receive the notification
        assert mock_send.call_count == 2
        calls_uris = [call.args[0] for call in mock_send.call_args_list]
        assert all("manifest" in uri for uri in calls_uris)

    def test_no_notification_when_no_subscribers(self):
        from unittest.mock import MagicMock, patch

        from trelix_mcp.subscriptions import SubscriptionRegistry, notify_file_changed

        reg = SubscriptionRegistry()  # no subscribers

        mock_send = MagicMock()
        with patch("trelix_mcp.subscriptions.send_resource_notification", mock_send):
            notify_file_changed(registry=reg, repo_path="/path", changed_file="src/auth.py")

        mock_send.assert_not_called()

    def test_docstrings_do_not_claim_delivery_or_an_unserved_request(self) -> None:
        """The registry docs must not call resources/subscribe served or trelix watch a pusher."""
        import trelix_mcp.subscriptions as subscriptions

        module_doc = subscriptions.__doc__ or ""
        hook_doc = subscriptions.notify_file_changed.__doc__ or ""
        docs = module_doc + hook_doc

        assert "Client → server:  resources/subscribe" not in docs
        assert "When trelix watch detects a file change" not in docs
        assert "Called by the MCP server's watch bridge" not in docs
        assert "nothing is delivered today" in module_doc
        assert "nothing is delivered today" in hook_doc


class TestMCPSubscriptionCapability:
    """trelix-mcp reports only what it serves: resources.subscribe is false.

    The SDK derives ServerCapabilities from the registered request handlers, and
    trelix-mcp registers no resources/subscribe (or subscriptions/listen) handler.
    These tests use the real server, in-process, the way a client sees it.
    """

    @pytest.mark.parametrize(
        ("mode", "is_handshake_era"),
        [("legacy", True), ("auto", False)],
    )
    async def test_server_does_not_advertise_resource_subscribe(
        self, mode: str, is_handshake_era: bool
    ) -> None:
        """Both protocol eras (initialize handshake, server/discover) report subscribe false."""
        from fastmcp import Client
        from trelix_mcp.server import mcp

        async with Client(mcp, mode=mode) as client:
            caps = client.server_capabilities
            assert (client.initialize_result is not None) == is_handshake_era

        assert caps is not None
        assert caps.resources is not None, "trelix-mcp registers resources, so the block must exist"
        assert caps.resources.subscribe is False

    @pytest.mark.parametrize("resources_changed", [True, False])
    def test_subscribe_is_false_whatever_list_changed_is(self, resources_changed: bool) -> None:
        """subscribe and listChanged are independent fields; neither switches the other on."""
        from mcp.server.lowlevel.server import NotificationOptions
        from trelix_mcp.server import mcp

        caps = mcp._mcp_server.get_capabilities(
            NotificationOptions(resources_changed=resources_changed),
            {},
        )

        assert caps.resources is not None
        assert caps.resources.subscribe is False

    def test_get_capabilities_is_not_patched(self) -> None:
        """Reported capabilities equal FastMCP's own derivation, for the call shapes mcp 2.x uses.

        Guards against re-installing a wrapper on the server instance that forces
        resources.subscribe=True. The unbound class method is the un-patched derivation.
        """
        from mcp.server.lowlevel.server import NotificationOptions
        from trelix_mcp.server import mcp

        server = mcp._mcp_server
        options = NotificationOptions(resources_changed=True)

        assert server.get_capabilities(options, {}) == type(server).get_capabilities(
            server, options, {}
        )
        assert server.get_capabilities(protocol_version="2026-07-28") == type(
            server
        ).get_capabilities(server, protocol_version="2026-07-28")

    @pytest.mark.filterwarnings("ignore:resources/subscribe is removed:UserWarning")
    async def test_resources_subscribe_request_is_not_served(self) -> None:
        """A raw resources/subscribe request is method-not-found, matching the capability."""
        from fastmcp import Client
        from mcp import MCPError
        from trelix_mcp.server import mcp

        async with Client(mcp, mode="legacy") as client:
            with pytest.raises(MCPError) as excinfo:
                await client.session.subscribe_resource("trelix://index/stats")

        assert excinfo.value.code == -32601

    async def test_subscribe_tool_description_matches_the_capability(self) -> None:
        """tools/list, the text models read, must not say to wait for resources.subscribe=True.

        The description used to tell models to call subscribe_resource "after receiving
        resources.subscribe=True in capabilities"; the server reports false now and delivers
        nothing. Whitespace is normalised so re-wrapping the docstring does not matter.
        """
        from fastmcp import Client
        from trelix_mcp.server import mcp

        async with Client(mcp, mode="legacy") as client:
            tools = await client.list_tools()

        tool = next(t for t in tools if t.name == "subscribe_resource")
        description = " ".join((tool.description or "").split())

        assert "resources.subscribe=True" not in description
        assert "does not advertise resources.subscribe" in description
        assert "none is delivered yet" in description


class TestSendResourceNotificationWireFormat:
    """Verify the exact JSON-RPC wire format written to stdout by send_resource_notification.

    MCP spec 2024-11-05 §Resources — notifications/resources/updated MUST carry:
      - jsonrpc = "2.0"
      - method  = "notifications/resources/updated"
      - params.uri            (the resource URI — only field in params)
      - params._meta.subscriptionId  (client correlation ID)
    No content payload — the client calls resources/read on demand.
    """

    def test_jsonrpc_version_is_2_0(self, capsys):
        import json

        from trelix_mcp.subscriptions import send_resource_notification

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="sub-abc",
        )
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["jsonrpc"] == "2.0", f"Expected jsonrpc=2.0, got {msg.get('jsonrpc')!r}"

    def test_method_is_notifications_resources_updated(self, capsys):
        import json

        from trelix_mcp.subscriptions import send_resource_notification

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="sub-abc",
        )
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["method"] == "notifications/resources/updated", (
            f"Expected method=notifications/resources/updated, got {msg.get('method')!r}"
        )

    def test_params_uri_matches_input(self, capsys):
        import json

        from trelix_mcp.subscriptions import send_resource_notification

        uri = "trelix://repo//Users/sai/myrepo/manifest"
        send_resource_notification(uri=uri, subscription_id="sub-xyz")
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["params"]["uri"] == uri, (
            f"params.uri mismatch: expected {uri!r}, got {msg['params'].get('uri')!r}"
        )

    def test_params_meta_contains_subscription_id(self, capsys):
        import json

        from trelix_mcp.subscriptions import send_resource_notification

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="my-sub-id-123",
        )
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert "_meta" in msg["params"], "params._meta missing from notification"
        assert msg["params"]["_meta"]["subscriptionId"] == "my-sub-id-123", (
            f"subscriptionId mismatch: {msg['params']['_meta'].get('subscriptionId')!r}"
        )

    def test_params_contains_no_content_payload(self, capsys):
        """Notifications must carry URI only — no content, text, or blob fields."""
        import json

        from trelix_mcp.subscriptions import send_resource_notification

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="sub-abc",
        )
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        params_keys = set(msg["params"].keys())
        forbidden = params_keys - {"uri", "_meta"}
        assert not forbidden, (
            f"params must contain only uri and _meta; unexpected keys: {forbidden}"
        )

    def test_output_is_valid_json_line(self, capsys):
        """Each notification must be a single valid JSON line terminated by newline."""
        import json

        from trelix_mcp.subscriptions import send_resource_notification

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="sub-abc",
        )
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.split("\n") if ln.strip()]
        assert len(lines) == 1, f"Expected exactly 1 JSON line, got {len(lines)}"
        # Must parse without error
        json.loads(lines[0])


class TestSendResourceNotificationConcurrency:
    def test_concurrent_notifications_do_not_interleave(self, capsys):
        """N threads calling send_resource_notification simultaneously must each
        produce exactly one intact JSON line — no torn/interleaved output."""
        import json
        import threading

        from trelix_mcp.subscriptions import send_resource_notification

        threads = [
            threading.Thread(
                target=send_resource_notification,
                args=(f"trelix://repo/r{i}/manifest", f"sub-{i}"),
            )
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.split("\n") if line]
        assert len(lines) == 20, f"Expected 20 output lines, got {len(lines)}"
        for line in lines:
            parsed = json.loads(line)  # must not raise — proves no torn/interleaved JSON
            assert parsed["method"] == "notifications/resources/updated"


class TestSubscriptionRegistryMaxCap:
    def test_subscribe_raises_when_at_capacity(self):
        from trelix_mcp.subscriptions import (
            SubscriptionLimitExceeded,
            SubscriptionRegistry,
        )

        reg = SubscriptionRegistry(max_subscribers=2)
        reg.subscribe("trelix://repo/a/manifest", "sub-1")
        reg.subscribe("trelix://repo/b/manifest", "sub-2")

        with pytest.raises(SubscriptionLimitExceeded):
            reg.subscribe("trelix://repo/c/manifest", "sub-3")

    def test_subscribe_allows_re_subscribing_existing_id_at_capacity(self):
        """Re-subscribing an already-tracked subscription_id must not count
        as a new slot — it's an update, not growth."""
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry(max_subscribers=1)
        reg.subscribe("trelix://repo/a/manifest", "sub-1")
        # Same subscription_id, different URI — must succeed, not raise.
        reg.subscribe("trelix://repo/b/manifest", "sub-1")
        assert reg.get_uri("sub-1") == "trelix://repo/b/manifest"

    def test_unbounded_by_default(self):
        """Default construction (no max_subscribers) preserves today's
        unbounded behavior exactly."""
        from trelix_mcp.subscriptions import SubscriptionRegistry

        reg = SubscriptionRegistry()
        for i in range(500):
            reg.subscribe(f"trelix://repo/r{i}/manifest", f"sub-{i}")
        assert len(reg.all_uris()) == 500


class TestSubscriptionRegistryTTL:
    def test_expired_subscription_is_evicted_on_next_read(self):
        from unittest.mock import patch

        from trelix_mcp.subscriptions import SubscriptionRegistry

        fake_time = {"t": 1000.0}
        with patch("trelix_mcp.subscriptions.time.time", side_effect=lambda: fake_time["t"]):
            reg = SubscriptionRegistry(ttl_seconds=60.0)
            reg.subscribe("trelix://repo/a/manifest", "sub-1")
            assert reg.get_uri("sub-1") == "trelix://repo/a/manifest"

            fake_time["t"] += 61.0  # advance past the 60s TTL
            assert reg.get_uri("sub-1") is None
            assert reg.get_subscription_ids("trelix://repo/a/manifest") == []

    def test_expired_subscription_frees_capacity_slot(self):
        """A cap + TTL combination: an expired entry must not block new
        subscriptions once evicted."""
        from unittest.mock import patch

        from trelix_mcp.subscriptions import SubscriptionRegistry

        fake_time = {"t": 1000.0}
        with patch("trelix_mcp.subscriptions.time.time", side_effect=lambda: fake_time["t"]):
            reg = SubscriptionRegistry(max_subscribers=1, ttl_seconds=60.0)
            reg.subscribe("trelix://repo/a/manifest", "sub-1")

            fake_time["t"] += 61.0
            # sub-1 has expired; this must succeed rather than raise
            # SubscriptionLimitExceeded, because the lazy sweep inside
            # subscribe() should evict it first.
            reg.subscribe("trelix://repo/b/manifest", "sub-2")
            assert reg.get_uri("sub-2") == "trelix://repo/b/manifest"

    def test_no_ttl_by_default(self):
        """Default construction (no ttl_seconds) preserves today's
        never-expires behavior exactly."""
        from unittest.mock import patch

        from trelix_mcp.subscriptions import SubscriptionRegistry

        fake_time = {"t": 1000.0}
        with patch("trelix_mcp.subscriptions.time.time", side_effect=lambda: fake_time["t"]):
            reg = SubscriptionRegistry()
            reg.subscribe("trelix://repo/a/manifest", "sub-1")
            fake_time["t"] += 10_000_000.0  # far future
            assert reg.get_uri("sub-1") == "trelix://repo/a/manifest"


class TestSubscribeResourceToolCapacityHandling:
    def test_subscribe_resource_tool_returns_error_dict_at_capacity(self, monkeypatch):
        """The MCP tool must convert SubscriptionLimitExceeded into an error
        dict, not propagate an uncaught exception to the client."""
        monkeypatch.setenv("TRELIX_MCP_MAX_SUBSCRIBERS", "1")
        import importlib

        import trelix_mcp.server as srv

        importlib.reload(srv)
        try:
            first = srv.subscribe_resource("trelix://repo/a/manifest", "sub-1")
            assert first["subscribed"] is True

            second = srv.subscribe_resource("trelix://repo/b/manifest", "sub-2")
            assert second["subscribed"] is False
            assert "capacity" in second["error"]
        finally:
            monkeypatch.delenv("TRELIX_MCP_MAX_SUBSCRIBERS", raising=False)
            importlib.reload(srv)
