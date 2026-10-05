"""
MCP resource subscription registry.

Tracks which subscription IDs are registered for which trelix:// resource URIs.
The registry is filled by the subscribe_resource / unsubscribe_resource tools in
server.py. trelix-mcp does not serve the resources/subscribe request (a client that
sends it gets "Method not found") and reports resources.subscribe as false.

notify_file_changed() looks up the subscription IDs for a repo's manifest URI and
writes notifications/resources/updated to stdout for each. That can only reach an MCP
client when it runs inside the stdio server process. The FileWatcher hook calls it
from the `trelix watch` process, where this registry is empty, and nothing starts a
watcher inside the server, so nothing is delivered today.

Notification wire format (MCP spec 2024-11-05 §Resources):
  1. Server → client:  notifications/resources/updated  { uri }  (URI only — no content)
  2. Client → server:  resources/read  { uri }  (client fetches updated content on demand)

Multiplexed over stdio's single bidirectional channel using subscriptionId in _meta.
"""

from __future__ import annotations

import json
import sys
import threading
import time


class SubscriptionLimitExceeded(Exception):
    """Raised when SubscriptionRegistry.subscribe() would exceed max_subscribers."""


class SubscriptionRegistry:
    """Thread-safe registry mapping resource URIs to MCP subscription IDs.

    max_subscribers and ttl_seconds both default to None (unbounded, never
    expires) — identical to the pre-existing behavior. Setting either
    bounds unchecked growth from a client that subscribes repeatedly
    without ever unsubscribing.
    """

    def __init__(
        self,
        max_subscribers: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        # uri -> set of subscription_ids
        self._uri_to_ids: dict[str, set[str]] = {}
        # subscription_id -> uri (reverse index for unsubscribe)
        self._id_to_uri: dict[str, str] = {}
        # subscription_id -> created-at timestamp (time.time()), for TTL sweeps
        self._created_at: dict[str, float] = {}
        self._max_subscribers = max_subscribers
        self._ttl_seconds = ttl_seconds

    def subscribe(self, uri: str, subscription_id: str) -> None:
        """Register a subscription ID as watching the given URI.

        Raises SubscriptionLimitExceeded if max_subscribers is set and the
        registry is already at capacity with a DIFFERENT subscription_id
        (re-subscribing an existing ID never counts as growth).
        """
        with self._lock:
            self._evict_expired_locked()
            at_capacity = (
                self._max_subscribers is not None
                and len(self._id_to_uri) >= self._max_subscribers
                and subscription_id not in self._id_to_uri
            )
            if at_capacity:
                raise SubscriptionLimitExceeded(
                    f"Cannot add subscription {subscription_id!r}: registry is "
                    f"at max capacity ({self._max_subscribers})"
                )
            # If this subscription_id was already watching a different URI,
            # detach it from that URI first.
            old_uri = self._id_to_uri.get(subscription_id)
            if old_uri is not None and old_uri != uri:
                self._uri_to_ids[old_uri].discard(subscription_id)
                if not self._uri_to_ids[old_uri]:
                    del self._uri_to_ids[old_uri]
            self._uri_to_ids.setdefault(uri, set()).add(subscription_id)
            self._id_to_uri[subscription_id] = uri
            self._created_at[subscription_id] = time.time()

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its ID."""
        with self._lock:
            uri = self._id_to_uri.pop(subscription_id, None)
            self._created_at.pop(subscription_id, None)
            if uri and uri in self._uri_to_ids:
                self._uri_to_ids[uri].discard(subscription_id)
                if not self._uri_to_ids[uri]:
                    del self._uri_to_ids[uri]

    def get_subscription_ids(self, uri: str) -> list[str]:
        """Return all subscription IDs currently watching the given URI."""
        with self._lock:
            self._evict_expired_locked()
            return list(self._uri_to_ids.get(uri, set()))

    def get_uri(self, subscription_id: str) -> str | None:
        """Return the URI a subscription ID is watching, or None if not found."""
        with self._lock:
            self._evict_expired_locked()
            return self._id_to_uri.get(subscription_id)

    def all_uris(self) -> list[str]:
        """Return all URIs with active subscribers."""
        with self._lock:
            self._evict_expired_locked()
            return list(self._uri_to_ids.keys())

    def _evict_expired_locked(self) -> None:
        """Remove subscriptions older than ttl_seconds. Caller must hold self._lock."""
        if self._ttl_seconds is None:
            return
        now = time.time()
        expired = [
            sub_id
            for sub_id, created in self._created_at.items()
            if now - created > self._ttl_seconds
        ]
        for sub_id in expired:
            uri = self._id_to_uri.pop(sub_id, None)
            self._created_at.pop(sub_id, None)
            if uri and uri in self._uri_to_ids:
                self._uri_to_ids[uri].discard(sub_id)
                if not self._uri_to_ids[uri]:
                    del self._uri_to_ids[uri]


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

    Only looks in the registry it is given. FileWatcher._do_reindex calls it in the
    `trelix watch` process, whose registry is empty, so nothing is delivered today.
    The manifest URI for a repo is: trelix://repo/{repo_path}/manifest
    """
    manifest_uri = f"trelix://repo/{repo_path}/manifest"
    subscription_ids = registry.get_subscription_ids(manifest_uri)
    for sid in subscription_ids:
        send_resource_notification(manifest_uri, sid)
