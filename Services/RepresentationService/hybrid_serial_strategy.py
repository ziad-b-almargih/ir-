import numpy as np

from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class HybridSerialStrategy(IRepresentationStrategy):
    """Cascade: a fast first model shortlists candidates, a smarter second model re-ranks them.

    Doesn't own an index of its own. It composes two already-built strategies (typically
    BM25 -> Embedding) and is constructed on demand by SearchService.
    """

    def __init__(self, first: IRepresentationStrategy, second: IRepresentationStrategy,
                 shortlist_k: int = 100):
        self.first = first
        self.second = second
        self.shortlist_k = shortlist_k

    def index_documents(self, raw_documents) -> None:
        # Nothing to build: relies on the children's existing indexes.
        pass

    def get_scores(self, query: str) -> np.ndarray:
        # Stage 1: cheap lexical scoring shortlists the top candidates.
        first_scores = np.asarray(self.first.get_scores(query))
        shortlist = np.argsort(first_scores)[::-1][:self.shortlist_k]

        # Stage 2: smarter scoring re-ranks ONLY the shortlist; others get -inf so they fall out.
        second_scores = np.asarray(self.second.get_scores(query))
        out = np.full_like(first_scores, -np.inf, dtype=np.float64)
        out[shortlist] = second_scores[shortlist]
        return out
