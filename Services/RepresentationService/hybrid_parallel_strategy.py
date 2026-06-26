from typing import List

import numpy as np

from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class HybridParallelStrategy(IRepresentationStrategy):
    """Reciprocal Rank Fusion (RRF): each child model ranks the corpus on its own,
    then every doc is scored by sum_i 1 / (k + rank_i). Position-based, so it ignores
    each model's score scale and stays robust when scales differ wildly (BM25 ~10, cosine ~0.7).
    """

    def __init__(self, strategies: List[IRepresentationStrategy], k: int = 60):
        self.strategies = strategies
        self.k = k

    def index_documents(self, raw_documents) -> None:
        # Nothing to build: relies on the children's existing indexes.
        pass

    def get_scores(self, query: str) -> np.ndarray:
        fused = None
        for strategy in self.strategies:
            scores = np.asarray(strategy.get_scores(query))
            if fused is None:
                fused = np.zeros(len(scores), dtype=np.float64)
            # Convert scores to 1-based ranks (best doc = rank 1) without sorting twice.
            desc_order = np.argsort(scores)[::-1]
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[desc_order] = np.arange(len(scores))
            ranks += 1
            fused += 1.0 / (self.k + ranks)
        return fused
