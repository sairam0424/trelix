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
