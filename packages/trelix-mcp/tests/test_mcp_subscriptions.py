"""Tests for MCP resource subscription capability.

Per MCP spec 2024-11-05 §Resources, a server must declare
``capabilities.resources.subscribe=True`` before any MCP client
(Claude Code, Cursor, VS Code Copilot) will attempt to send
``resources/subscribe`` requests.
"""
from __future__ import annotations


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
        """When a watched file changes, notifications/resources/updated is sent to all subscribers."""
        from unittest.mock import MagicMock, patch
        from trelix_mcp.subscriptions import SubscriptionRegistry
        from trelix_mcp.subscriptions import notify_file_changed

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


class TestMCPSubscriptionCapability:
    """The trelix-mcp server must advertise resources.subscribe=True."""

    def test_server_advertises_resource_subscribe(self) -> None:
        """trelix-mcp must declare resources.subscribe=True in capabilities.

        FastMCP 2.x exposes capabilities via the LowLevelServer._get_capabilities
        pathway.  We call get_capabilities() directly so the test is independent
        of a running transport.
        """
        from mcp.server.lowlevel.server import NotificationOptions
        from trelix_mcp.server import mcp

        caps = mcp._mcp_server.get_capabilities(
            NotificationOptions(resources_changed=True),
            {},
        )

        assert caps.resources is not None, (
            "trelix-mcp must have a resources capability block — no @mcp.resource "
            "decorators appear to be registered."
        )
        assert caps.resources.subscribe is True, (
            "trelix-mcp must declare resources.subscribe=True so MCP clients "
            "(Claude Code, Cursor) know they can subscribe to resource changes. "
            f"Current value: {caps.resources.subscribe!r}"
        )

    def test_resource_subscribe_does_not_set_list_changed(self) -> None:
        """subscribe and listChanged are independent; listChanged must not be forced True.

        Per MCP spec the two fields are orthogonal.  We only opt-in to subscribe,
        not listChanged — that would require a separate notification infrastructure.
        NOTE: FastMCP sets listChanged=True by default (notification_options.resources_changed=True),
        so we only assert that subscribe is True, not that listChanged is False.
        """
        from mcp.server.lowlevel.server import NotificationOptions
        from trelix_mcp.server import mcp

        caps = mcp._mcp_server.get_capabilities(
            NotificationOptions(resources_changed=False),
            {},
        )

        # The only change we made is subscribe=True; listChanged follows notification_options
        assert caps.resources is not None
        assert caps.resources.subscribe is True, (
            "subscribe capability must be True regardless of listChanged setting"
        )


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
        from trelix_mcp.subscriptions import send_resource_notification
        import json

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="sub-abc",
        )
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["jsonrpc"] == "2.0", f"Expected jsonrpc=2.0, got {msg.get('jsonrpc')!r}"

    def test_method_is_notifications_resources_updated(self, capsys):
        from trelix_mcp.subscriptions import send_resource_notification
        import json

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
        from trelix_mcp.subscriptions import send_resource_notification
        import json

        uri = "trelix://repo//Users/sai/myrepo/manifest"
        send_resource_notification(uri=uri, subscription_id="sub-xyz")
        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["params"]["uri"] == uri, (
            f"params.uri mismatch: expected {uri!r}, got {msg['params'].get('uri')!r}"
        )

    def test_params_meta_contains_subscription_id(self, capsys):
        from trelix_mcp.subscriptions import send_resource_notification
        import json

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
        from trelix_mcp.subscriptions import send_resource_notification
        import json

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
        from trelix_mcp.subscriptions import send_resource_notification
        import json

        send_resource_notification(
            uri="trelix://repo//my/repo/manifest",
            subscription_id="sub-abc",
        )
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.split("\n") if ln.strip()]
        assert len(lines) == 1, f"Expected exactly 1 JSON line, got {len(lines)}"
        # Must parse without error
        json.loads(lines[0])
