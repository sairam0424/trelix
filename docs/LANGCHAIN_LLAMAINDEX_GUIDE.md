# LangChain & LlamaIndex Integration Guide

This guide covers integrating Trelix with LangChain and LlamaIndex to build
code-aware RAG pipelines over any repository.

---

## LangChain Integration

### Install

```bash
pip install trelix-langchain==2.4.0
pip install "trelix[local]"
```

### TrelixRetriever

`TrelixRetriever` is a LangChain-compatible retriever that wraps Trelix's hybrid
search (dense vector + BM25 + call-graph) and returns standard `Document` objects.

```python
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(
    repo_path="/path/to/repo",
    provider="local",  # or "openai", "azure", "voyage"
    k=10,
)
```

#### Constructor Parameters

| Parameter   | Type   | Default   | Description                                      |
|-------------|--------|-----------|--------------------------------------------------|
| `repo_path` | `str`  | required  | Absolute path to the repository root             |
| `provider`  | `str`  | `"local"` | Embedding provider: `local`, `openai`, `azure`, `voyage` |
| `k`         | `int`  | `10`      | Number of results to return per query            |

### Use in a RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo")
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    "Use the following code context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("how does the authentication middleware work?")
print(answer)
```

### Document Format

Each retrieved `Document` object contains:

| Field                        | Type    | Description                              |
|------------------------------|---------|------------------------------------------|
| `page_content`               | `str`   | Full source code of the symbol           |
| `metadata.source`            | `str`   | Relative file path (`rel_path`)          |
| `metadata.symbol`            | `str`   | Fully qualified symbol name              |
| `metadata.language`          | `str`   | Language: `python`, `typescript`, `go`, etc. |
| `metadata.kind`              | `str`   | Symbol kind: `function`, `class`, `method` |
| `metadata.lines`             | `str`   | Line range, e.g. `"10-25"`              |
| `metadata.score`             | `float` | Relevance score in range `0.0` – `1.0`  |
| `metadata.retrieval_source`  | `str`   | Source pipeline: `vector`, `bm25`, `call_graph` |

### Full API Reference

```python
class TrelixRetriever(BaseRetriever):
    """LangChain retriever backed by Trelix hybrid search."""

    repo_path: str
    provider: str = "local"
    k: int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Run hybrid search and return k Documents."""
        ...

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Async variant of _get_relevant_documents."""
        ...
```

---

## LlamaIndex Integration

### Install

```bash
pip install trelix-llama-index==2.4.0
```

### TrelixIndexRetriever

`TrelixIndexRetriever` is a LlamaIndex-compatible retriever that returns
`NodeWithScore` objects suitable for use in any LlamaIndex query engine or
pipeline.

```python
from trelix_llama_index import TrelixIndexRetriever

retriever = TrelixIndexRetriever(
    repo_path="/path/to/repo",
    provider="local",
    k=10,
)
```

#### Constructor Parameters

| Parameter   | Type   | Default   | Description                                      |
|-------------|--------|-----------|--------------------------------------------------|
| `repo_path` | `str`  | required  | Absolute path to the repository root             |
| `provider`  | `str`  | `"local"` | Embedding provider: `local`, `openai`, `azure`, `voyage` |
| `k`         | `int`  | `10`      | Number of nodes to return per query              |

### Use in a Query Engine

```python
from trelix_llama_index import TrelixIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = TrelixIndexRetriever(
    repo_path="/path/to/repo",
    provider="local",
    k=10,
)

query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
response = query_engine.query("how does authentication work?")
print(response.response)
```

### Node Format

Each `NodeWithScore` object contains:

| Field                 | Type    | Description                         |
|-----------------------|---------|-------------------------------------|
| `node.text`           | `str`   | Full source code of the symbol      |
| `node.metadata.file`  | `str`   | Relative file path                  |
| `node.metadata.symbol`| `str`   | Fully qualified symbol name         |
| `score`               | `float` | Relevance score in range `0.0` – `1.0` |

### Advanced: Custom RAG Pipeline

The example below wires a custom `ServiceContext` with a local LLM and
Trelix retrieval for a fully offline code-QA pipeline.

```python
from trelix_llama_index import TrelixIndexRetriever
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Configure a local LLM and embedding model
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Build the Trelix retriever
retriever = TrelixIndexRetriever(
    repo_path="/path/to/repo",
    provider="local",
    k=8,
)

# 3. Build a response synthesizer
synthesizer = get_response_synthesizer(response_mode="compact")

# 4. Assemble the query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)

# 5. Query
response = query_engine.query(
    "Walk me through the request lifecycle from the HTTP handler to the database."
)

# 6. Inspect source nodes
print(response.response)
print()
for node in response.source_nodes:
    print(f"  {node.node.metadata['file']}  ({node.node.metadata['symbol']})  score={node.score:.3f}")
```

#### Expected Output (truncated)

```
The request enters via handle_request() in src/server/handler.py, which
validates the incoming payload with RequestSchema before handing off to
RouterMiddleware...

  src/server/handler.py  (handle_request)  score=0.923
  src/middleware/router.py  (RouterMiddleware.dispatch)  score=0.871
  src/db/session.py  (get_session)  score=0.804
```

---

## Provider Reference

Both `TrelixRetriever` and `TrelixIndexRetriever` accept the same `provider`
values.

| Value     | Package required                         | Notes                       |
|-----------|------------------------------------------|-----------------------------|
| `local`   | `trelix[local]` (default, no API key)    | Uses a local sentence-transformer model |
| `openai`  | `trelix[openai]`                         | Requires `OPENAI_API_KEY`   |
| `azure`   | `trelix[azure]`                          | Requires Azure OpenAI env vars |
| `voyage`  | `trelix[voyage]`                         | Requires `VOYAGE_API_KEY`   |

---

## Common Patterns

### Filtering by language

```python
# LangChain — post-filter Documents
docs = retriever.get_relevant_documents("parse tokens")
python_docs = [d for d in docs if d.metadata["language"] == "python"]
```

```python
# LlamaIndex — post-filter NodeWithScore
nodes = retriever.retrieve("parse tokens")
python_nodes = [n for n in nodes if n.node.metadata.get("language") == "python"]
```

### Displaying sources

```python
# LangChain
for doc in docs:
    print(f"{doc.metadata['source']}:{doc.metadata['lines']}  ({doc.metadata['symbol']})")
```

```python
# LlamaIndex
for n in nodes:
    print(f"{n.node.metadata['file']}  ({n.node.metadata['symbol']})  score={n.score:.3f}")
```

---

## Troubleshooting

**`ModuleNotFoundError: trelix_langchain`** — install `trelix-langchain`, not
`trelix-langchan` (common typo).

**Slow first query** — the `local` provider downloads the embedding model on
first use (~90 MB). Subsequent queries are fast.

**Empty results** — run `trelix index /path/to/repo` to build the index before
querying. Both integrations require a pre-built Trelix index.

**Score always 0.0 in LlamaIndex** — update to `trelix-llama-index>=2.4.0`;
earlier versions did not propagate scores from the hybrid ranker.
