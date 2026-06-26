import numpy as np
from rank_bm25 import BM25Okapi

from Services.PreProcessingService.preprocessing_service import PreprocessingService
from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class Bm25Strategy(IRepresentationStrategy):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # k1 controls term-frequency saturation; b controls document-length normalization.
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self._preprocessor = None

    @property
    def preprocessor(self) -> PreprocessingService:
        # Built lazily and kept out of the pickle (see __getstate__) so saved indexes stay small.
        if self._preprocessor is None:
            self._preprocessor = PreprocessingService()
        return self._preprocessor

    def index_documents(self, raw_documents) -> None:
        # Preprocess each raw document on the fly (streamed, so memory stays low).
        tokenized = (self.preprocessor.process_text(text) for text in raw_documents)
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def get_scores(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            raise ValueError("Documents not indexed yet. Call index_documents first.")
        tokens = self.preprocessor.process_text(query)
        return np.asarray(self.bm25.get_scores(tokens))

    def get_scores_with_params(self, query: str, k1: float, b: float) -> np.ndarray:
        # Re-score with custom k1/b WITHOUT rebuilding the index: the per-document stats
        # (term frequencies, idf, lengths, avgdl) are already stored, only k1/b plug in here.
        # This is what lets the UI tune BM25 parameters per query.
        if self.bm25 is None:
            raise ValueError("Documents not indexed yet. Call index_documents first.")
        tokens = self.preprocessor.process_text(query)
        bm = self.bm25
        doc_len = np.array(bm.doc_len)
        scores = np.zeros(bm.corpus_size)
        for term in tokens:
            term_freqs = np.array([doc.get(term, 0) for doc in bm.doc_freqs])
            idf = bm.idf.get(term, 0)
            scores += idf * (term_freqs * (k1 + 1) /
                             (term_freqs + k1 * (1 - b + b * doc_len / bm.avgdl)))
        return scores

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_preprocessor"] = None
        return state
