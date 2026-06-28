import numpy as np

from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class HybridWeightedStrategy(IRepresentationStrategy):
    """Score-level fusion with an explicit alpha:  final = alpha * lex + (1-alpha) * sem.

    Each child's scores are min-max normalized per query so BM25 (~10) and cosine (~0.7)
    sit on the same [0, 1] scale before being combined. alpha=1.0 = pure lexical,
    alpha=0.0 = pure semantic, alpha=0.5 = equal blend.
    """

    def __init__(self, lexical: IRepresentationStrategy, semantic: IRepresentationStrategy,
                 alpha: float = 0.5):
        self.lexical = lexical
        self.semantic = semantic
        self.alpha = float(alpha)

    def index_documents(self, raw_documents) -> None:
        # Composed strategy: relies on the children's already-built indexes.
        pass

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def get_scores(self, query: str) -> np.ndarray:
        lex = self._normalize(self.lexical.get_scores(query))
        sem = self._normalize(self.semantic.get_scores(query))
        return self.alpha * lex + (1.0 - self.alpha) * sem
