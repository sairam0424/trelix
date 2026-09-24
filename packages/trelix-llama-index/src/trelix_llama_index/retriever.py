from typing import TYPE_CHECKING

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

if TYPE_CHECKING:
    from trelix.retrieval.retriever import Retriever


class TrelixIndexRetriever(BaseRetriever):
    def __init__(self, repo_path: str, provider: str = "local", k: int = 10) -> None:
        self._repo_path = repo_path
        self._provider = provider
        self._k = k
        # Memoized across calls on this instance. Retriever construction is
        # expensive -- for the `local` embedder specifically, make_embedder()
        # loads a SentenceTransformer model from disk, several seconds every
        # time -- and repo_path/provider are fixed for the lifetime of this
        # instance, so there is nothing to invalidate.
        self._cached_retriever: Retriever | None = None
        super().__init__()

    def _get_trelix_retriever(self) -> "Retriever":
        if self._cached_retriever is not None:
            return self._cached_retriever

        from typing import Literal, cast

        from trelix.core.config import EmbedderConfig, IndexConfig
        from trelix.retrieval.retriever import Retriever

        config = IndexConfig(
            repo_path=self._repo_path,
            embedder=EmbedderConfig(
                provider=cast(
                    Literal[
                        "openai",
                        "azure",
                        "local",
                        "voyage",
                        "local-code",
                        "bedrock-titan",
                        "bedrock-cohere",
                        "bge-code",
                        "nomic-code",
                        "cohere",
                    ],
                    self._provider,
                )
            ),
        )
        self._cached_retriever = Retriever(config)
        return self._cached_retriever

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
