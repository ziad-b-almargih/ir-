import numpy as np
from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class RepresentationContext:
    def __init__(self, strategy: IRepresentationStrategy = None):
        self._strategy = strategy

    def set_strategy(self, strategy: IRepresentationStrategy):
        # Swap the active representation strategy at runtime (Strategy pattern).
        self._strategy = strategy

    def index_documents(self, documents) -> None:
        if not self._strategy:
            raise ValueError("Strategy not set. Cannot index documents.")
        self._strategy.index_documents(documents)

    def get_scores(self, query: str) -> np.ndarray:
        if not self._strategy:
            raise ValueError("Strategy not set. Cannot score query.")
        return self._strategy.get_scores(query)
