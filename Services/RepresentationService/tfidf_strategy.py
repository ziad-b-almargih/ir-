import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Services.PreProcessingService.preprocessing_service import PreprocessingService
from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


def dummy_processor(doc):
    # Identity callback: documents/queries arrive already tokenized, so skip sklearn's own tokenizing.
    return doc


class TfIdfStrategy(IRepresentationStrategy):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_processor,
            preprocessor=dummy_processor,
            token_pattern=None,
            lowercase=False
        )
        self.tfidf_matrix = None
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
        self.tfidf_matrix = self.vectorizer.fit_transform(tokenized)

    def get_scores(self, query: str) -> np.ndarray:
        if self.tfidf_matrix is None:
            raise ValueError("Documents not indexed yet. Call index_documents first.")
        tokens = self.preprocessor.process_text(query)
        query_vector = self.vectorizer.transform([tokens])
        return cosine_similarity(query_vector, self.tfidf_matrix).flatten()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_preprocessor"] = None
        return state
