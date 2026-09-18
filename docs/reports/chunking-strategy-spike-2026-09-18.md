# Chunking-strategy spike — 2026-09-18

**Phase 6-style research spike.** No code ships from the main finding below; one small, independently-justified fix is scoped separately at the end.

## Question

trelix's `Chunker.build_chunks()` (`src/trelix/indexing/chunker.py`) builds exactly one `Chunk` per parsed `Symbol` (function/method/class) — this is the "Function" chunking strategy in the taxonomy of a controlled empirical study that found per-symbol chunking underperforms every alternative tested. Does that finding mean trelix should change its chunking strategy?

## What the research found (verified via a 94-agent adversarial-verification research pass; 19/25 claims confirmed, 6 refuted)

**Source:** arXiv:2605.04763, Wu/Gong/Jahangirova/Zhang (King's College London, submitted May 2026), "How Does Chunking Affect Retrieval-Augmented Code Completion? A Controlled Empirical Study" — 864 experimental settings across 4 chunking strategies × 4 retrievers × 5 generators × 9 parameter configs, on RepoEval and CrossCodeEval.

- **Confirmed, high confidence:** Function/per-symbol chunking (trelix's current approach) underperforms Declaration (signature-only), Sliding Window (fixed-size line windows), and cAST (AST split+merge) by 3.57–5.64pp Exact Match on RepoEval, Cliff's δ=−1.0 — a fully consistent effect across every retriever/generator/split combination tested. Function chunking is never Pareto-optimal on the cost-quality frontier.
- **Confirmed, high confidence, important nuance:** the margin between the three *non*-Function strategies is narrow (0.38–2.07pp) — Sliding Window edges out overall, but cAST actually wins on the Java subset of CrossCodeEval. This is not "Function is bad, X is the clear winner" — it's "Function is bad, the other three are roughly tied."
- **Confirmed, high confidence, most load-bearing finding for trelix:** cross-file context length — not chunking strategy — is the single most influential parameter the study found. Doubling the context window from 2,048→8,192 tokens produced up to 4.2pp EM improvement, exceeding even retriever choice; chunk size itself had a weaker, non-monotonic effect (≤1.9pp).
- **Confirmed:** cAST's own paper (arXiv:2506.15655, CMU/Augment Code, EMNLP 2025) separately reports +4.3 Recall@5 (RepoEval retrieval) and +2.67 Pass@1 (SWE-bench), varying by retriever backbone (not uniform).
- **Confirmed:** the paper's own Discussion (§6.4) proposes structure-aware chunking as a *context-compression* mechanism (what to keep/discard) rather than as the retrieval unit itself — an open future-work direction, no follow-up found yet (paper is ~4.5 months old).
- **Refuted / unverifiable — do not treat as fact:** specific claims that Cursor uses sibling-merging or that Greptile embeds one docstring per AST node both failed independent verification (0-3 votes). Only Continue.dev has verifiable published guidance, and it explicitly recommends *against* full AST-based chunking by default (framed as "most exact, but most complex," not worth the cost for most users) — and doesn't have a dedicated one-symbol-per-chunk mode to "move away from" in the first place. The premise that production RAG code tools have already moved off per-symbol chunking for this reason is **not substantiated**.

## What this means for trelix specifically

trelix is not a bare RAG-completion pipeline of the kind this study benchmarks — it already has two mechanisms that address exactly the gap the study attributes to narrow per-symbol chunks, but at the *retrieval* layer instead of the *chunking* layer:

1. **Multi-leg RRF fusion + call-graph/dataflow expansion** (`expand_with_call_graph`, `expand_with_dataflow`) already pull in a symbol's callers/callees/dataflow-adjacent code at query time, which is a different route to the same goal (giving the model context beyond one function's own body) that the study's context-length finding says matters most.
2. **`ContextualChunker`** already prepends an LLM-generated summary to each chunk, which is a different mechanism than cAST's structural merge but addresses a similar "isolated chunk lacks context" problem.

Given this, and given the narrow, non-decisive margin between the three winning strategies plus the unverified state of any real production adoption of full AST split+merge chunking, adopting cAST's sibling-merge wholesale is a large, cross-cutting change for an uncertain, possibly-already-partially-captured gain: it would break the current 1-chunk-to-1-symbol cardinality assumed throughout `SearchResult`, the reranker reconstruction sites, and `graph_context` attribution — the exact class of "silently drops a field across reconstruction sites" bug this session already found three separate times in adjacent code this same week.

**Verdict: LOW-MEDIUM confidence that a full AST split+merge chunking redesign is worth building right now.** Not because the underlying research is weak (it's well-verified) — because the trelix-specific case for it is diluted by (a) already-partially-overlapping existing mechanisms, (b) narrow margins among the actual alternatives, and (c) a real, structurally-similar-to-recent-bugs implementation risk for an uncertain gain.

## One independently-justified, low-risk fix worth shipping regardless

Reading `_truncate_chunk()` in `chunker.py` surfaced a real, separate gap: symbols exceeding `max_tokens_per_chunk` (default 512) are currently **destructively truncated** — the excess body is discarded outright, not split into a second chunk. This is the "recursive split" half of cAST's design (independent of the "merge" half, and independent of the win/loss margins above) — and it's a strict improvement with no cardinality change (still N chunks trace back cleanly to symbols, just possibly >1 chunk per oversized symbol instead of exactly 1). Recommending this as a follow-up task, scoped to `Chunker.build_chunks`/`_truncate_chunk` only.
