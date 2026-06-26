from typing import Dict, List, Optional

import numpy as np
import psycopg

from Services.DocumentStoreService.document_store_service import (
    DocumentStoreService, _conn_kwargs)
from Services.IndexingService.indexing_service import IndexingService


class TopicService:
    """Discovers topical clusters with BERTopic on top of the existing embeddings.

    Offline: fit BERTopic on a sample for speed, then assign a topic to every document
    using the precomputed embedding matrix. Persist topics + per-doc assignments to Postgres.
    Online: list topics, or filter search to documents within a chosen topic.
    """

    def __init__(self):
        self._conn = None

    def _connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(**_conn_kwargs(), autocommit=True)
            self.ensure_schema()
        return self._conn

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    dataset  TEXT NOT NULL,
                    topic_id INTEGER NOT NULL,
                    label    TEXT,
                    words    TEXT[],
                    size     INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (dataset, topic_id)
                );
                CREATE TABLE IF NOT EXISTS doc_topics (
                    dataset  TEXT NOT NULL,
                    doc_id   TEXT NOT NULL,
                    topic_id INTEGER NOT NULL,
                    PRIMARY KEY (dataset, doc_id)
                );
                CREATE INDEX IF NOT EXISTS doc_topics_topic_idx ON doc_topics (dataset, topic_id);
                """
            )

    # --------------------------- offline: build topics ---------------------------
    def build(self, dataset_name: str, sample_size: int = 50000, seed: int = 42) -> Dict:
        from bertopic import BERTopic

        indexing = IndexingService()
        corpus = indexing.load_corpus(dataset_name)
        embedding_strategy = indexing.load_index(dataset_name, "embedding")
        embeddings = np.asarray(embedding_strategy.doc_embeddings, dtype=np.float32)
        doc_ids = corpus.doc_ids
        n_total = len(doc_ids)

        # Sample for fitting (UMAP/HDBSCAN on 522K is heavy; sample is plenty for topics).
        sample_size = min(sample_size, n_total)
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_total, sample_size, replace=False)
        sample_idx.sort()

        # Sample raw texts for c-TF-IDF (which words best describe each topic).
        docstore = DocumentStoreService()
        sample_ids = [doc_ids[i] for i in sample_idx]
        text_map = docstore.get_texts(dataset_name, sample_ids)
        sample_texts = [text_map.get(doc_id, "") for doc_id in sample_ids]
        sample_embeddings = embeddings[sample_idx]

        print(f"[TopicService] Fitting BERTopic on sample={sample_size} ...")
        topic_model = BERTopic(verbose=True, calculate_probabilities=False)
        topic_model.fit(sample_texts, embeddings=sample_embeddings)

        # Assign a topic to EVERY document using its precomputed embedding (no re-encoding).
        # BERTopic.transform requires `documents` for a shape check only; placeholder strings
        # are enough. We chunk to keep peak memory low.
        print(f"[TopicService] Transforming all {n_total} embeddings ...")
        all_topics = np.empty(n_total, dtype=np.int64)
        chunk = 50_000
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            placeholder_docs = [""] * (end - start)
            chunk_topics, _ = topic_model.transform(
                documents=placeholder_docs, embeddings=embeddings[start:end])
            all_topics[start:end] = np.asarray(chunk_topics, dtype=np.int64)
            print(f"[TopicService] transformed {end}/{n_total}", flush=True)

        # Persist: topics table (id, label, words, size) and doc_topics (doc_id -> topic_id).
        print("[TopicService] Saving topics and assignments ...")
        info = topic_model.get_topic_info()           # one row per topic
        size_counts = self._counts(all_topics)

        with self._connection().cursor() as cur:
            cur.execute("DELETE FROM topics WHERE dataset = %s", (dataset_name,))
            for _, row in info.iterrows():
                topic_id = int(row["Topic"])
                words = [w for w, _ in topic_model.get_topic(topic_id)] if topic_id != -1 else []
                label = row.get("Name", str(topic_id))
                size = int(size_counts.get(topic_id, 0))
                cur.execute(
                    "INSERT INTO topics (dataset, topic_id, label, words, size) VALUES (%s,%s,%s,%s,%s)",
                    (dataset_name, topic_id, label, words, size),
                )

            cur.execute("DELETE FROM doc_topics WHERE dataset = %s", (dataset_name,))
            with cur.copy("COPY doc_topics (dataset, doc_id, topic_id) FROM STDIN") as copy:
                for doc_id, topic_id in zip(doc_ids, all_topics):
                    copy.write_row((dataset_name, doc_id, int(topic_id)))

        print(f"[TopicService] Done. {len(info)} topics, {n_total} doc assignments saved.")
        return {"topics": len(info), "documents": n_total, "sample_used": sample_size}

    # --------------------------- online queries ---------------------------
    def list_topics(self, dataset_name: str) -> List[Dict]:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT topic_id, label, words, size FROM topics "
                "WHERE dataset = %s ORDER BY size DESC",
                (dataset_name,),
            )
            return [
                {"topic_id": tid, "label": label, "words": list(words or []), "size": size}
                for tid, label, words, size in cur.fetchall()
            ]

    def doc_ids_for_topic(self, dataset_name: str, topic_id: int) -> List[str]:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT doc_id FROM doc_topics WHERE dataset = %s AND topic_id = %s",
                (dataset_name, topic_id),
            )
            return [doc_id for (doc_id,) in cur.fetchall()]

    @staticmethod
    def _counts(topics) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for topic_id in topics:
            counts[int(topic_id)] = counts.get(int(topic_id), 0) + 1
        return counts
