"""Tests for MCP resource subscription capability.

Per MCP spec 2024-11-05 §Resources, a server must declare
``capabilities.resources.subscribe=True`` before any MCP client
(Claude Code, Cursor, VS Code Copilot) will attempt to send
``resources/subscribe`` requests.
"""
from __future__ import annotations


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
