# trelix-langchain

LangChain retriever for [trelix](https://github.com/sairam0424/trelix) — semantic code search using Tree-sitter AST parsing, hybrid BM25+vector search, and call-graph expansion.

## Install

```bash
pip install trelix-langchain
```

For AWS Bedrock embeddings (Cohere or Titan):

```bash
pip install "trelix-langchain[bedrock]"
```

## Basic Usage

```python
from trelix_langchain import TrelixRetriever

# First index your repo (one-time)
# trelix index /path/to/repo

retriever = TrelixRetriever(repo_path="/path/to/repo", k=10)
docs = retriever.invoke("how does authentication work?")

for doc in docs:
    print(doc.metadata["source"], doc.metadata["score"])
    print(doc.page_content[:200])
```

Each returned `Document` carries rich metadata:

| Metadata key | Example value |
|---|---|
| `source` | `"src/auth/middleware.py"` |
| `symbol` | `"auth.middleware.require_login"` |
| `language` | `"python"` |
| `kind` | `"function"` |
| `lines` | `"42-78"` |
| `score` | `0.91` |
| `retrieval_source` | `"hybrid"` |

## LangChain RAG Chain (LCEL)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo", k=8)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the code context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(
        f"# {d.metadata['source']} ({d.metadata['symbol']})\n{d.page_content}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = chain.invoke("How does the authentication middleware work?")
print(answer)
```

## RetrievalQA (classic interface)

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo", k=10)
llm = ChatOpenAI(model="gpt-4o")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

result = qa.invoke({"query": "Where is rate limiting applied?"})
print(result["result"])
for doc in result["source_documents"]:
    print(" -", doc.metadata["source"])
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider: `local` \| `local-code` \| `openai` \| `azure` \| `voyage` \| `bedrock-cohere` \| `bedrock-titan` |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `AZURE_API_KEY` | — | Required for `azure` provider |
| `AWS_ACCESS_KEY_ID` | — | Required for Bedrock providers |
| `AWS_SECRET_ACCESS_KEY` | — | Required for Bedrock providers |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock |

You can also set the provider directly on the retriever instance:

```python
retriever = TrelixRetriever(repo_path="/path/to/repo", provider="openai", k=10)
```

## Provider Switching (v0.7.0+)

```bash
# Use Bedrock Cohere embeddings (best retrieval quality, reuses AWS credentials)
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere trelix index /path/to/repo

# Use Azure OpenAI embeddings
TRELIX_EMBEDDER_PROVIDER=azure trelix index /path/to/repo

# Use local sentence-transformers (no API key needed, works offline)
TRELIX_EMBEDDER_PROVIDER=local trelix index /path/to/repo
```

The index and the retriever must use the same provider — re-index whenever you switch.

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
- [trelix-mcp](https://pypi.org/project/trelix-mcp/) — MCP server for Claude Code, Cursor, Windsurf
- [trelix-llama-index](https://pypi.org/project/trelix-llama-index/) — LlamaIndex retriever
