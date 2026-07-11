"""
MCP resource subscription registry.

Tracks which MCP subscription IDs are watching which trelix:// resource URIs.
When trelix watch detects a file change, it looks up all subscription IDs for
the affected URI and sends notifications/resources/updated for each.

Wire protocol (MCP spec 2024-11-05 §Resources, confirmed 3-0 adversarial):
  1. Client → server:  resources/subscribe  { uri }
  2. Server → client:  notifications/resources/updated  { uri }  (URI only — no content)
  3. Client → server:  resources/read  { uri }  (client fetches updated content on demand)

Multiplexed over stdio's single bidirectional channel using subscriptionId in _meta.
"""

from __future__ import annotations

import json
import sys
import threading


class SubscriptionRegistry:
    """Thread-safe registry mapping resource URIs to MCP subscription IDs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # uri -> set of subscription_ids
        self._uri_to_ids: dict[str, set[str]] = {}
        # subscription_id -> uri (reverse index for unsubscribe)
        self._id_to_uri: dict[str, str] = {}

    def subscribe(self, uri: str, subscription_id: str) -> None:
        """Register a subscription ID as watching the given URI."""
        with self._lock:
            self._uri_to_ids.setdefault(uri, set()).add(subscription_id)
            self._id_to_uri[subscription_id] = uri

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its ID."""
        with self._lock:
            uri = self._id_to_uri.pop(subscription_id, None)
            if uri and uri in self._uri_to_ids:
                self._uri_to_ids[uri].discard(subscription_id)
                if not self._uri_to_ids[uri]:
                    del self._uri_to_ids[uri]

    def get_subscription_ids(self, uri: str) -> list[str]:
        """Return all subscription IDs currently watching the given URI."""
        with self._lock:
            return list(self._uri_to_ids.get(uri, set()))

    def get_uri(self, subscription_id: str) -> str | None:
        """Return the URI a subscription ID is watching, or None if not found."""
        with self._lock:
            return self._id_to_uri.get(subscription_id)

    def all_uris(self) -> list[str]:
        """Return all URIs with active subscribers."""
        with self._lock:
            return list(self._uri_to_ids.keys())


# Serializes writes to stdout so overlapping calls to send_resource_notification
# (e.g. FileWatcher's threading.Timer callback firing on multiple debounced
# file changes close together) cannot interleave partial JSON lines.
_stdout_lock = threading.Lock()


def send_resource_notification(uri: str, subscription_id: str) -> None:
    """Write a notifications/resources/updated JSON-RPC message to stdout.

    The MCP spec (2024-11-05 §Resources) specifies notifications/resources/updated
    carries only the URI — no content. The client calls resources/read on demand.

    The subscriptionId is placed in _meta for client-side correlation.
    This multiplexes over stdio's single bidirectional channel.
    """
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {
            "uri": uri,
            "_meta": {"subscriptionId": subscription_id},
        },
    }
    line = json.dumps(notification) + "\n"
    # MCP stdio: each message is a JSON line terminated by \n
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def notify_file_changed(
    registry: SubscriptionRegistry,
    repo_path: str,
    changed_file: str,
) -> None:
    """Fire notifications/resources/updated for all subscribers watching repo_path.

    Called by the MCP server's watch bridge when watchfiles detects a change.
    The manifest URI for a repo is: trelix://repo/{repo_path}/manifest
    """
    manifest_uri = f"trelix://repo/{repo_path}/manifest"
    subscription_ids = registry.get_subscription_ids(manifest_uri)
    for sid in subscription_ids:
        send_resource_notification(manifest_uri, sid)
