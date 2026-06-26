
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class TrelixRetriever(BaseRetriever):
    repo_path: str
    provider: str = "local"
    k: int = 10

    def _get_trelix_retriever(self):
        from typing import Literal, cast

        from trelix.core.config import EmbedderConfig, IndexConfig
        from trelix.retrieval.retriever import Retriever

        config = IndexConfig(
            repo_path=self.repo_path,
            embedder=EmbedderConfig(
                provider=cast(
                    Literal["openai", "azure", "local", "voyage", "local-code"],
                    self.provider,
                )
            ),
        )
        return Retriever(config)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        ctx = self._get_trelix_retriever().retrieve(query)
        return [
            Document(
                page_content=r.symbol.body,
                metadata={
                    "source": r.file.rel_path,
                    "symbol": r.symbol.qualified_name,
                    "language": r.file.language.value,
                    "kind": r.symbol.kind.value,
                    "lines": str(r.symbol.line_start) + "-" + str(r.symbol.line_end),
                    "score": r.score,
                    "retrieval_source": r.source,
                },
            )
            for r in ctx.results[: self.k]
        ]
