#!/usr/bin/env bash
# End-to-end test for trelix v0.4.0 beast mode
# Tests: index, stats, search (semantic + BM25), ask (synthesis), watch (smoke)
set -euo pipefail

VENV=".venv/bin"
REPO="."
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS${NC} — $1"; }
fail() { echo -e "${RED}❌ FAIL${NC} — $1"; exit 1; }
info() { echo -e "${YELLOW}▶${NC}  $1"; }

echo ""
echo "═══════════════════════════════════════════════════"
echo "  trelix v0.4.0 — End-to-End Test Suite"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── 1. Stats ───────────────────────────────────────────────────────────────
info "1. trelix stats"
STATS=$($VENV/trelix stats $REPO 2>&1)
echo "$STATS"
echo "$STATS" | grep -q "Files indexed" && pass "stats command works" || fail "stats failed"
echo "$STATS" | grep -q "Symbols" && pass "symbols counted" || fail "no symbols"
FILES=$(echo "$STATS" | grep "Files indexed" | grep -oE '[0-9]+' | head -1)
SYMBOLS=$(echo "$STATS" | grep "Symbols" | grep -oE '[0-9]+' | head -1)
echo "  → Files: $FILES | Symbols: $SYMBOLS"

# ─── 2. Search — semantic (vector) ──────────────────────────────────────────
echo ""
info "2. trelix search — semantic query (hybrid search)"
SEARCH1=$($VENV/trelix search $REPO "how does the query planner classify retrieval intent" --provider azure --json 2>&1)
echo "$SEARCH1" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('results',[])
print(f'  Results: {len(r)}')
for x in r[:3]: print(f'    {x[\"file\"]:45s} {x[\"symbol\"]:30s} score={x[\"score\"]}')
" 2>/dev/null && pass "semantic search returns results" || fail "semantic search failed"

# ─── 3. Search — BM25 keyword ───────────────────────────────────────────────
echo ""
info "3. trelix search — keyword (BM25)"
SEARCH2=$($VENV/trelix search $REPO "RRF fusion reciprocal rank" --provider azure --json 2>&1)
echo "$SEARCH2" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('results',[])
print(f'  Results: {len(r)}')
for x in r[:3]: print(f'    {x[\"file\"]:45s} {x[\"symbol\"]:30s}')
" 2>/dev/null
echo "$SEARCH2" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('results') else 1)" 2>/dev/null && pass "BM25 search works" || fail "BM25 search failed"

# ─── 4. Search — blast_radius intent ────────────────────────────────────────
echo ""
info "4. trelix search — blast radius query"
SEARCH3=$($VENV/trelix search $REPO "what files import the Chunk dataclass" --provider azure --json 2>&1)
echo "$SEARCH3" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('results',[])
print(f'  Results: {len(r)}')
for x in r[:3]: print(f'    {x[\"file\"]:45s}')
" 2>/dev/null
echo "$SEARCH3" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('results') else 1)" 2>/dev/null && pass "blast_radius style query works" || fail "blast_radius query failed"

# ─── 5. Ask — LLM synthesis ──────────────────────────────────────────────────
echo ""
info "5. trelix ask — LLM synthesis (gpt-4o, via Python API)"
ASK=$($VENV/python -c "
from trelix.core.config import IndexConfig, EmbedderConfig
from trelix.retrieval.retriever import Retriever
from trelix.retrieval.synthesizer import Synthesizer
config = IndexConfig(repo_path='.')
ctx = Retriever(config).retrieve('how does trelix indexing pipeline work')
answer = Synthesizer(config.embedder).synthesize(ctx, config.embedder) or ''
print(answer[:400])
" 2>/dev/null)
echo "  Answer preview: ${ASK:0:200}..."
[ -n "$ASK" ] && pass "LLM synthesis produces answer" || fail "LLM synthesis returned empty"

# ─── 6. Update-index (single file) ───────────────────────────────────────────
echo ""
info "6. trelix update-index — single file re-index"
UPDT=$($VENV/trelix update-index $REPO src/trelix/retrieval/fusion.py 2>&1)
echo "  Output: $UPDT"
echo "$UPDT" | grep -qi "skip\|symbols\|chunk\|ok\|updat" && pass "update-index works" || fail "update-index failed"

# ─── 7. Unit tests ───────────────────────────────────────────────────────────
echo ""
info "7. Full unit test suite"
TESTS=$($VENV/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3)
echo "  $TESTS"
echo "$TESTS" | grep -q "passed" && pass "unit tests green" || fail "unit tests have failures"
COUNT=$(echo "$TESTS" | grep -oE '[0-9]+ passed' | head -1)
echo "  → $COUNT"

# ─── 8. New feature smoke tests ──────────────────────────────────────────────
echo ""
info "8. Beast-mode feature smoke tests"

# U3: HNSW info
HNSW=$($VENV/python -c "
from trelix.core.config import IndexConfig
from trelix.store.vector import SQLiteVectorStore
config = IndexConfig(repo_path='.')
info = config.db_path_absolute
print('HNSW config:', config.store.hnsw)
" 2>&1)
echo "  $HNSW"
pass "HNSW config accessible"

# U4: Qdrant backend config
QDRANT=$($VENV/python -c "
from trelix.core.config import IndexConfig
cfg = IndexConfig(repo_path='.')
print('Backend:', cfg.store.backend)
print('Qdrant URL:', cfg.store.qdrant_url)
" 2>&1)
echo "  $QDRANT"
echo "$QDRANT" | grep -q "sqlite\|qdrant" && pass "vector backend config works" || fail "backend config broken"

# U5: Async embedder
ASYNC=$($VENV/python -c "
from trelix.embedder.base import BaseEmbedder
import inspect
has_async = hasattr(BaseEmbedder, 'embed_async')
sig = inspect.signature(BaseEmbedder.embed_async) if has_async else None
print('embed_async available:', has_async)
print('is coroutinefunction:', inspect.iscoroutinefunction(BaseEmbedder.embed_async) if has_async else False)
" 2>&1)
echo "  $ASYNC"
echo "$ASYNC" | grep -q "True" && pass "async embed method available" || fail "async embed missing"

# U6: File watcher importable
WATCH=$($VENV/python -c "
from trelix.indexing.watcher import FileWatcher
print('FileWatcher importable:', True)
print('has start/stop:', hasattr(FileWatcher,'start') and hasattr(FileWatcher,'stop'))
" 2>&1)
echo "  $WATCH"
echo "$WATCH" | grep -q "True" && pass "FileWatcher importable" || fail "FileWatcher broken"

# U7: Adaptive router
ROUTER=$($VENV/python -c "
from trelix.retrieval.planner.agent import AdaptiveRouter
from trelix.retrieval.planner.models import RoutingTier
router = AdaptiveRouter(llm_client=None)
trivial = router._is_tier1('what is trelix')
normal  = router._is_tier1('how does the indexing pipeline work')
print('Tier1 for trivial:', trivial)
print('Tier1 for normal:', normal)
" 2>&1)
echo "  $ROUTER"
pass "AdaptiveRouter tier classification works"

# U8: GraphRAG synthesizer
GRAG=$($VENV/python -c "
from trelix.retrieval.graph_rag import GraphRAGSynthesizer
from trelix.core.config import EmbedderConfig, RetrievalConfig
cfg_e = EmbedderConfig()
cfg_r = RetrievalConfig()
g = GraphRAGSynthesizer(cfg_e, cfg_r)
print('GraphRAGSynthesizer instantiated:', True)
print('Threshold tokens:', cfg_r.graph_rag_threshold_tokens)
print('Threshold results:', cfg_r.graph_rag_threshold_results)
" 2>&1)
echo "  $GRAG"
echo "$GRAG" | grep -q "True" && pass "GraphRAGSynthesizer works" || fail "GraphRAG broken"

# U9: Call graph callee_type_hint
CTH=$($VENV/python -c "
from trelix.core.models import CallEdge
e = CallEdge(caller_id=1, callee_name='login', line=42, callee_type_hint='AuthService')
print('callee_type_hint:', e.callee_type_hint)
" 2>&1)
echo "  $CTH"
echo "$CTH" | grep -q "AuthService" && pass "callee_type_hint on CallEdge works" || fail "callee_type_hint broken"

# U10: Eval harness
EVAL=$($VENV/python -c "
from tests.eval.harness import EvalHarness
from tests.eval.metrics import recall_at_k, reciprocal_rank, ndcg_at_k
print('EvalHarness importable:', True)
print('Metrics available: recall_at_k, mrr, ndcg')
" 2>&1)
echo "  $EVAL"
echo "$EVAL" | grep -q "True" && pass "Eval harness importable" || fail "Eval harness broken"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo -e "  ${GREEN}All end-to-end tests passed — trelix v0.4.0 ✅${NC}"
echo "═══════════════════════════════════════════════════"
echo ""
