from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class TrelixIndexRetriever(BaseRetriever):
    def __init__(self, repo_path, provider="local", k=10):
        self._repo_path = repo_path
        self._provider = provider
        self._k = k
        super().__init__()

    def _get_trelix_retriever(self):
        from trelix.core.config import EmbedderConfig, IndexConfig
        from trelix.retrieval.retriever import Retriever
        from typing import Literal, cast

        config = IndexConfig(
            repo_path=self._repo_path,
            embedder=EmbedderConfig(
                provider=cast(
                    Literal["openai", "azure", "local", "voyage", "local-code"],
                    self._provider,
                )
            ),
        )
        return Retriever(config)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        ctx = self._get_trelix_retriever().retrieve(query_bundle.query_str)
        return [
            NodeWithScore(
                node=TextNode(
                    text=r.symbol.body,
                    metadata={
                        "file": r.file.rel_path,
                        "symbol": r.symbol.qualified_name,
                    },
                ),
                score=r.score,
            )
            for r in ctx.results[: self._k]
        ]
