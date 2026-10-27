"""
Golden file format v2: the optional entry fields and the values they may take.

v2 is v1 plus optional keys. A line still carries `query` and `relevant_files` exactly as
`trelix.eval.harness` describes, and may add:

    id           label for the query, unique within the file
    lang         language of the code the query is about
    kind         how the query was phrased or found: nl | keyword | commit | issue
    source       where the query came from
    gold_status  the result of checking relevant_files: validated | pooled | unreviewed
    split        dev | test

Every one is optional. A file that uses none of them is a v1 file and loads and scores as it
always did, and keys this module does not know stay allowed. `validate_entry` type-checks the
fields a line does carry; the loader calls it, so a mistyped `kind` is refused with its line
number like every other unusable entry, and `trelix eval-validate` builds on it.
"""

from __future__ import annotations

from collections.abc import Mapping

KINDS = ("nl", "keyword", "commit", "issue")
GOLD_STATUSES = ("validated", "pooled", "unreviewed")
SPLITS = ("dev", "test")

# The `gold_status` values that count toward `trelix eval-validate --min-validated`. A tuple,
# not a set: the value compared against it comes from a JSON file and may be unhashable.
REVIEWED_GOLD_STATUSES = ("validated", "pooled")

_TEXT_FIELDS = ("id", "lang", "source")
_CHOICE_FIELDS = {"kind": KINDS, "gold_status": GOLD_STATUSES, "split": SPLITS}


def validate_entry(item: Mapping[str, object]) -> list[str]:
    """Describe what is wrong with the v2 fields `item` carries; `[]` when they are fine.

    Only fields that are present are checked, and `null` counts as present: a `kind` of
    `null` is a wrong-typed `kind`, not an absent one. `query` and `relevant_files` are
    not looked at here; the loader owns those.
    """
    problems: list[str] = []
    for name in _TEXT_FIELDS:
        if name not in item:
            continue
        value = item[name]
        if not isinstance(value, str) or not value.strip():
            problems.append(f'"{name}" must be a non-empty string')
    for name, allowed in _CHOICE_FIELDS.items():
        if name in item and item[name] not in allowed:
            problems.append(f'"{name}" must be one of {", ".join(allowed)} (got {item[name]!r})')
    return problems
