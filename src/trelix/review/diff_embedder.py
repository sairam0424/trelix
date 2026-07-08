"""
Semantic diff embeddings — CCRep-style before/after code body pairs.

Reference: CCRep (ICSE 2023, arXiv:2302.03924): encode a code change as the
concatenation of before-change and after-change code bodies, fed into a
pre-trained code model to produce contextual embeddings.

Enables 'historically similar diffs' retrieval in trelix review --pr:
  1. At review time, embed each PR hunk (before+after bodies)
  2. Search stored diff_chunks for similar past changes
  3. Surface: 'This change looks like the auth fix in PR #23'

Storage: diff_chunks SQLite table (added to db.py schema).
Chunking: hunk-granular with MAX_DIFF_CHUNKS=500 cap and MAX_EMBED_CHARS
truncation for SVG blobs and minified JS (validated: chunkhound PR #288).
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.embedder.base import BaseEmbedder
    from trelix.store.db import Database

logger = logging.getLogger("trelix.review.diff_embedder")

# Max chars to embed per hunk (before+after concatenated).
# Prevents pathological SVG/minified JS from dominating embedding budget.
MAX_EMBED_CHARS = 2000
MAX_DIFF_CHUNKS = 500


class DiffEmbedder:
    """Embed and store code diff hunks for similarity retrieval."""

    MAX_EMBED_CHARS = MAX_EMBED_CHARS

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def embed_hunk(self, before_code: str, after_code: str) -> list[float]:
        """
        Embed a code change as a before+after body pair (CCRep encoding).

        Concatenates before and after bodies with a separator, truncates to
        MAX_EMBED_CHARS, and embeds using the configured embedder.

        Returns the embedding vector. Never raises — returns [] on failure.
        """
        try:
            combined = f"{before_code}\n---\n{after_code}"
            if len(combined) > MAX_EMBED_CHARS:
                combined = combined[:MAX_EMBED_CHARS]
            return self._embedder.embed_query(combined)
        except Exception as exc:
            logger.debug("DiffEmbedder.embed_hunk failed: %s", exc)
            return []

    def store_pr_diff(
        self,
        db: Database,
        pr_ref: str,
        hunks: list[dict],
    ) -> int:
        """
        Embed and store all hunks for a PR reference.

        Each hunk dict must have: {hunk_header, before_code, after_code}.
        Caps at MAX_DIFF_CHUNKS hunks per PR.

        Returns number of chunks stored.
        """
        stored = 0
        for hunk in hunks[:MAX_DIFF_CHUNKS]:
            before = hunk.get("before_code", "")
            after = hunk.get("after_code", "")
            header = hunk.get("hunk_header", "")
            char_count = len(before) + len(after)

            embedding = self.embed_hunk(before_code=before, after_code=after)
            if not embedding:
                continue

            try:
                packed = struct.pack(f"{len(embedding)}f", *embedding)
                db._conn.execute(
                    """INSERT INTO diff_chunks
                       (pr_ref, hunk_header, before_code, after_code,
                        embedding, chunk_char_count)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (pr_ref, header, before, after, packed, char_count),
                )
                stored += 1
            except Exception as exc:
                logger.debug("Failed to store diff chunk: %s", exc)

        if stored:
            db._conn.commit()
        return stored

    def search_similar_diffs(
        self,
        db: Database,
        query_before: str,
        query_after: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Find historically similar diffs using before+after embedding similarity.

        Returns list of {pr_ref, hunk_header, before_code, after_code, score}
        sorted descending by cosine similarity.
        """
        import math

        query_emb = self.embed_hunk(before_code=query_before, after_code=query_after)
        if not query_emb:
            return []

        rows = db._conn.execute(
            "SELECT pr_ref, hunk_header, before_code, after_code, embedding "
            "FROM diff_chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        results = []
        q_norm = math.sqrt(sum(v * v for v in query_emb)) or 1.0

        for pr_ref, header, before, after, packed in rows:
            if not packed:
                continue
            try:
                n = len(packed) // 4
                stored_emb = list(struct.unpack(f"{n}f", packed))
                dot = sum(a * b for a, b in zip(query_emb, stored_emb))
                s_norm = math.sqrt(sum(v * v for v in stored_emb)) or 1.0
                score = dot / (q_norm * s_norm)
                results.append(
                    {
                        "pr_ref": pr_ref,
                        "hunk_header": header,
                        "before_code": before,
                        "after_code": after,
                        "score": score,
                    }
                )
            except Exception:
                continue

        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
