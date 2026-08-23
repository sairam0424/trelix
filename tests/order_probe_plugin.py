"""Opt-in order-dependence probes. INERT unless requested with ``-p``.

WHY THIS EXISTS RATHER THAN JUST pytest-randomly
------------------------------------------------
pytest-randomly's shuffle is HIERARCHICAL: it permutes module blocks, then
classes inside a module, then tests inside a class. Measured on tests/unit at
seeds 7, 424242 and 20260823, every run had
``contiguous-module-blocks == distinct-modules == 183``, i.e. no test from
module A ever ran between two tests of module B. So no seed can ever:

  * finalize a module-scoped fixture and re-create it mid-session, or
  * put another module's test between two tests of the same module.

That is exactly where this repo's order-dependent defects have lived. The one
found by this plugin (``tests/unit/test_otel_metrics.py``'s module-scoped
MeterProvider fixture, since moved to session scope) was invisible to all three
pytest-randomly seeds and to reverse order, and is reachable in production by
``pytest-xdist``, which hands individual tests to workers.

USAGE (both are full-suite probes, ~2-3 min on tests/unit; NOT for addopts)
--------------------------------------------------------------------------
    pytest tests/unit -p tests.order_probe_plugin --interleave --interleave-seed=1
    pytest tests/unit -p tests.order_probe_plugin --reverse-order

Each prints a CONTROL LINE that reads differently when the reordering silently
failed to apply, so a green run cannot be mistaken for a probe that never ran:

  --interleave    ``contiguous-module-blocks`` must be >> ``distinct-modules``
                  (measured 3823 vs 184). Equal means no interleaving happened.
  --reverse-order ``first=`` must equal the LAST id of the deterministic
                  collection order.
"""

from __future__ import annotations

import random
from collections import OrderedDict


def pytest_addoption(parser: object) -> None:
    group = parser.getgroup("order-probe")  # type: ignore[attr-defined]
    group.addoption(
        "--interleave",
        action="store_true",
        help="round-robin tests across modules so consecutive tests come from different files",
    )
    group.addoption(
        "--interleave-seed",
        action="store",
        type=int,
        default=0,
        help="seed for --interleave's module ordering",
    )
    group.addoption(
        "--reverse-order",
        action="store_true",
        help="run the exact reverse of the deterministic collection order",
    )


def _module_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _blocks(items: list) -> int:
    mods = [_module_of(i.nodeid) for i in items]
    return 1 + sum(1 for a, b in zip(mods, mods[1:]) if a != b)


def _distinct(items: list) -> int:
    return len({_module_of(i.nodeid) for i in items})


def pytest_collection_modifyitems(session: object, config: object, items: list) -> None:
    if config.getoption("--reverse-order"):  # type: ignore[attr-defined]
        items.reverse()
        print(
            f"\n[order-probe reverse] items={len(items)} first={items[0].nodeid} "
            f"contiguous-module-blocks={_blocks(items)} distinct-modules={_distinct(items)} "
            "(CONTROL: first= must equal the LAST id of the deterministic order)"
        )
        return

    if not config.getoption("--interleave"):  # type: ignore[attr-defined]
        return

    buckets: OrderedDict[str, list] = OrderedDict()
    for item in items:
        buckets.setdefault(_module_of(item.nodeid), []).append(item)

    rng = random.Random(config.getoption("--interleave-seed"))  # type: ignore[attr-defined]  # noqa: S311 - shuffling test order, not generating secrets
    keys = list(buckets)
    rng.shuffle(keys)

    out: list = []
    while keys:
        for key in list(keys):
            out.append(buckets[key].pop(0))
            if not buckets[key]:
                keys.remove(key)
    items[:] = out

    print(
        f"\n[order-probe interleave] seed={config.getoption('--interleave-seed')} "  # type: ignore[attr-defined]
        f"items={len(items)} contiguous-module-blocks={_blocks(items)} "
        f"distinct-modules={_distinct(items)} "
        "(CONTROL: blocks must be >> modules, else interleaving did NOT apply)"
    )
