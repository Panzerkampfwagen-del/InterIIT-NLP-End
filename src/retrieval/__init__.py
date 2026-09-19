"""Retrieval (Phase 6): hybrid search, incremental indexer, embeddings."""
from src.retrieval.embeddings import EMBEDDING_MODEL_VERSION, embed
from src.retrieval.indexer import RetrievalIndexer
from src.retrieval.query_builder import (
    RetrievalQuery,
    UnscopedQueryError,
    global_query,
    scoped_query,
)
from src.retrieval.search import HybridSearcher, SearchHit

__all__ = ["EMBEDDING_MODEL_VERSION", "embed", "RetrievalIndexer", "RetrievalQuery", "UnscopedQueryError", "global_query", "scoped_query", "HybridSearcher", "SearchHit"]
