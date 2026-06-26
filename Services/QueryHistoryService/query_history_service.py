import re
from typing import List, Tuple

import numpy as np
import psycopg

from Services.DocumentStoreService.document_store_service import _conn_kwargs

_WORD_RE = re.compile(r"[a-z]+")


class QueryHistoryService:
    """Stores past queries with their embeddings and uses them to enhance new queries.

    Two uses:
    - similar_queries: nearest past queries to surface as suggestions.
    - expansion_terms: words from those nearest queries to silently append before scoring.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._conn = None
        self._cache = None  # (texts, matrix) — invalidated on insert

        # Lazy import keeps cold-start cheap; reuse the project's English stopword list.
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        self._stop_words = set(stopwords.words('english'))

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(**_conn_kwargs(), autocommit=True)
            self.ensure_schema()
        return self._conn

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id         SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    embedding  BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

    def _encode(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

    def record(self, query: str) -> None:
        query = (query or "").strip()
        if not query:
            return
        embedding = self._encode(query)
        with self._connection().cursor() as cur:
            cur.execute(
                "INSERT INTO query_history (query_text, embedding) VALUES (%s, %s)",
                (query, embedding.tobytes()),
            )
        self._cache = None  # next read reloads with the new row

    def _load_cache(self):
        if self._cache is not None:
            return self._cache
        with self._connection().cursor() as cur:
            cur.execute("SELECT query_text, embedding FROM query_history ORDER BY id")
            rows = cur.fetchall()
        if not rows:
            self._cache = ([], np.zeros((0, 0), dtype=np.float32))
            return self._cache
        texts = [text for text, _ in rows]
        matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        self._cache = (texts, matrix)
        return self._cache

    def similar_queries(self, query: str, k: int = 5,
                        min_sim: float = 0.30) -> List[Tuple[str, float]]:
        texts, matrix = self._load_cache()
        if not texts:
            return []
        qvec = self._encode(query)
        # matrix rows and qvec are L2-normalized, so dot product == cosine similarity.
        sims = matrix @ qvec

        out: List[Tuple[str, float]] = []
        seen = {query.strip().lower()}
        for idx in sims.argsort()[::-1]:
            sim = float(sims[idx])
            if sim < min_sim:
                break
            text = texts[idx]
            key = text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((text, sim))
            if len(out) >= k:
                break
        return out

    def expansion_terms(self, query: str, k_queries: int = 5,
                        max_terms: int = 4, min_sim: float = 0.35) -> List[str]:
        # Pick content words from the most similar past queries, weighted by similarity.
        neighbours = self.similar_queries(query, k=k_queries, min_sim=min_sim)
        if not neighbours:
            return []

        query_words = set(_WORD_RE.findall(query.lower()))
        scored: dict = {}
        for text, sim in neighbours:
            for word in _WORD_RE.findall(text.lower()):
                if len(word) < 3 or word in query_words or word in self._stop_words:
                    continue
                scored[word] = scored.get(word, 0.0) + sim

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [word for word, _ in ranked[:max_terms]]
