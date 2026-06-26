import numpy as np

from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class EmbeddingStrategy(IRepresentationStrategy):
    """Dense semantic representation using a sentence-transformers model.

    Documents and queries are encoded into normalized vectors, so cosine similarity
    reduces to a dot product. The heavy model itself is never pickled (see __getstate__);
    only the document embedding matrix is saved.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self.doc_embeddings = None  # shape (n_docs, dim), L2-normalized

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_documents(self, documents) -> None:
        # `documents` is an iterable of RAW text strings (the model does its own tokenizing).
        texts = list(documents)
        total = len(texts)

        # Encode in chunks so the log shows persistent milestones — if the process is
        # killed we can see exactly how far we got (the tqdm bar is buffered and lost).
        chunk_size = 50_000
        pieces = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            piece = self.model.encode(
                texts[start:end],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype(np.float32)
            pieces.append(piece)
            print(f"[Embedding] chunk done: {end}/{total} documents", flush=True)

        self.doc_embeddings = np.vstack(pieces)

    def get_scores(self, query: str) -> np.ndarray:
        if self.doc_embeddings is None:
            raise ValueError("Documents not indexed yet. Call index_documents first.")
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        # Both sides are normalized, so the dot product equals cosine similarity.
        return self.doc_embeddings @ query_vector

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None  # never pickle the heavy transformer model
        return state
