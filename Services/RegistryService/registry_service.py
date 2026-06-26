import threading
from typing import Dict, List, Optional

import psycopg

from Services.DocumentStoreService.document_store_service import _conn_kwargs

# Status values a database entry can hold.
STATUS_BUILDING = "building"
STATUS_READY = "ready"
STATUS_ERROR = "error"


class RegistryService:
    """Tracks which datasets are loaded as indexes, in a Postgres table.

    Public API matches the previous JSON-backed version so the rest of the code is unchanged.
    """

    _lock = threading.Lock()

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
                CREATE TABLE IF NOT EXISTS databases (
                    dataset    TEXT PRIMARY KEY,
                    models     TEXT[]      NOT NULL DEFAULT '{}',
                    status     TEXT        NOT NULL DEFAULT '',
                    doc_count  INTEGER     NOT NULL DEFAULT 0,
                    error      TEXT        NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def list_databases(self) -> Dict[str, dict]:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT dataset, models, status, doc_count, error, "
                "to_char(updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS') "
                "FROM databases ORDER BY dataset"
            )
            rows = cur.fetchall()
        return {
            dataset: {
                "models": list(models),
                "status": status,
                "doc_count": doc_count,
                "error": error,
                "updated_at": updated_at,
            }
            for dataset, models, status, doc_count, error, updated_at in rows
        }

    def get(self, dataset: str) -> Optional[dict]:
        return self.list_databases().get(dataset)

    def ready_datasets(self) -> List[str]:
        return [name for name, info in self.list_databases().items()
                if info.get("status") == STATUS_READY]

    def upsert(self, dataset: str, models: Optional[List[str]] = None,
               status: Optional[str] = None, doc_count: Optional[int] = None,
               error: str = "") -> None:
        # Fetch-merge-write so partial updates keep existing fields. Lock keeps concurrent
        # background builds from clobbering each other; the UPSERT itself is atomic.
        with self._lock:
            conn = self._connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT models, status, doc_count FROM databases WHERE dataset = %s",
                    (dataset,),
                )
                row = cur.fetchone()
                existing_models, existing_status, existing_count = row if row else ([], "", 0)

                new_models = models if models is not None else list(existing_models)
                new_status = status if status is not None else existing_status
                new_count = doc_count if doc_count is not None else existing_count

                cur.execute(
                    """
                    INSERT INTO databases (dataset, models, status, doc_count, error, updated_at)
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (dataset) DO UPDATE SET
                        models     = EXCLUDED.models,
                        status     = EXCLUDED.status,
                        doc_count  = EXCLUDED.doc_count,
                        error      = EXCLUDED.error,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (dataset, new_models, new_status, new_count, error),
                )
