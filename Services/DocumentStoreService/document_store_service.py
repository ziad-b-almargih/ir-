import os
from typing import Dict, List

import psycopg

from Services.DataLoaderService.data_loader_service import DataLoaderService


def _conn_kwargs() -> dict:
    # Connection details come from environment variables (no secrets in code).
    return dict(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "ir_db"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "12345678"),
    )


class DocumentStoreService:
    """Stores raw document text in PostgreSQL and serves it by id at query time.

    Offline: bulk-load (dataset, doc_id, raw_text). Online: fetch the original raw text
    of the top results by id (a primary-key lookup, well within the 20s budget).
    """

    def __init__(self):
        self._conn = None

    def _connection(self):
        # Lazily open and reuse a single connection; reconnect if it dropped.
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(**_conn_kwargs(), autocommit=True)
        return self._conn

    def ensure_schema(self) -> None:
        with self._connection().cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    dataset  TEXT NOT NULL,
                    doc_id   TEXT NOT NULL,
                    raw_text TEXT,
                    PRIMARY KEY (dataset, doc_id)
                )
                """
            )

    def load_dataset(self, dataset_name: str) -> int:
        # Offline: (re)load every raw document for a dataset using fast COPY.
        self.ensure_schema()
        loader = DataLoaderService(dataset_name)
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE dataset = %s", (dataset_name,))
            count = 0
            with cur.copy("COPY documents (dataset, doc_id, raw_text) FROM STDIN") as copy:
                for document in loader.iter_documents():
                    copy.write_row((dataset_name, document.doc_id, document.text))
                    count += 1
        print(f"[DocumentStore] Loaded {count} documents for '{dataset_name}'.")
        return count

    def get_texts(self, dataset_name: str, doc_ids: List[str]) -> Dict[str, str]:
        # Online: fetch raw text for the given ids in one indexed query.
        if not doc_ids:
            return {}
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT doc_id, raw_text FROM documents WHERE dataset = %s AND doc_id = ANY(%s)",
                (dataset_name, list(doc_ids)),
            )
            return {doc_id: text for doc_id, text in cur.fetchall()}

    def count(self, dataset_name: str) -> int:
        with self._connection().cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE dataset = %s", (dataset_name,))
            return cur.fetchone()[0]
