from abc import ABC, abstractmethod
import numpy as np


class IRepresentationStrategy(ABC):
    @abstractmethod
    def index_documents(self, documents) -> None:
        # Build and store internal state once. Input form is strategy-specific
        # (lexical models take tokenized docs, embeddings take raw text).
        pass

    @abstractmethod
    def get_scores(self, query: str) -> np.ndarray:
        # Take the RAW query string and return a relevance score per document,
        # shape (n_docs,). Each strategy preprocesses the query as it sees fit.
        pass
