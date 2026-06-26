from typing import Dict, List, Optional

import psycopg

from Services.DocumentStoreService.document_store_service import _conn_kwargs

STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_ERROR = "error"


class EvaluationStoreService:
    """Persists evaluation runs (one row per run) in Postgres so the UI can chart them."""

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
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id SERIAL PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    model TEXT NOT NULL,
                    refine BOOLEAN NOT NULL DEFAULT FALSE,
                    used_all_qrels BOOLEAN NOT NULL DEFAULT FALSE,
                    num_queries INTEGER NOT NULL DEFAULT 0,
                    queries_evaluated INTEGER NOT NULL DEFAULT 0,
                    map_score DOUBLE PRECISION,
                    recall_score DOUBLE PRECISION,
                    p10_score DOUBLE PRECISION,
                    ndcg10_score DOUBLE PRECISION,
                    elapsed_seconds DOUBLE PRECISION,
                    status TEXT NOT NULL DEFAULT 'ready',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    finished_at TIMESTAMPTZ
                )
                """
            )

    def create_running(self, dataset: str, model: str, refine: bool,
                       used_all_qrels: bool, num_queries: int) -> int:
        with self._connection().cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_runs
                  (dataset, model, refine, used_all_qrels, num_queries, status)
                VALUES (%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (dataset, model, refine, used_all_qrels, num_queries, STATUS_RUNNING),
            )
            return cur.fetchone()[0]

    def finalize(self, run_id: int, metrics: Dict) -> None:
        with self._connection().cursor() as cur:
            cur.execute(
                """
                UPDATE evaluation_runs
                SET queries_evaluated = %s,
                    map_score        = %s,
                    recall_score     = %s,
                    p10_score        = %s,
                    ndcg10_score     = %s,
                    elapsed_seconds  = %s,
                    status           = %s,
                    finished_at      = now()
                WHERE id = %s
                """,
                (
                    metrics.get("queries_evaluated", 0),
                    metrics.get("MAP"),
                    metrics.get("Recall@100"),
                    metrics.get("P@10"),
                    metrics.get("nDCG@10"),
                    metrics.get("elapsed_seconds"),
                    STATUS_READY,
                    run_id,
                ),
            )

    def fail(self, run_id: int, error: str) -> None:
        with self._connection().cursor() as cur:
            cur.execute(
                "UPDATE evaluation_runs SET status=%s, error=%s, finished_at=now() WHERE id=%s",
                (STATUS_ERROR, error, run_id),
            )

    def save_completed(self, dataset: str, model: str, refine: bool,
                       used_all_qrels: bool, metrics: Dict) -> int:
        # Convenience for synchronous runs: insert directly as 'ready'.
        with self._connection().cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_runs
                  (dataset, model, refine, used_all_qrels, num_queries, queries_evaluated,
                   map_score, recall_score, p10_score, ndcg10_score, elapsed_seconds,
                   status, finished_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                RETURNING id
                """,
                (
                    dataset, model, refine, used_all_qrels,
                    metrics.get("queries_used", 0),
                    metrics.get("queries_evaluated", 0),
                    metrics.get("MAP"),
                    metrics.get("Recall@100"),
                    metrics.get("P@10"),
                    metrics.get("nDCG@10"),
                    metrics.get("elapsed_seconds"),
                    STATUS_READY,
                ),
            )
            return cur.fetchone()[0]

    def list(self, dataset: Optional[str] = None) -> List[Dict]:
        with self._connection().cursor() as cur:
            if dataset:
                cur.execute(
                    """SELECT id, dataset, model, refine, used_all_qrels, num_queries,
                              queries_evaluated, map_score, recall_score, p10_score,
                              ndcg10_score, elapsed_seconds, status, error,
                              to_char(created_at AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS')
                       FROM evaluation_runs WHERE dataset=%s ORDER BY id DESC""",
                    (dataset,),
                )
            else:
                cur.execute(
                    """SELECT id, dataset, model, refine, used_all_qrels, num_queries,
                              queries_evaluated, map_score, recall_score, p10_score,
                              ndcg10_score, elapsed_seconds, status, error,
                              to_char(created_at AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS')
                       FROM evaluation_runs ORDER BY id DESC"""
                )
            keys = ["id", "dataset", "model", "refine", "used_all_qrels", "num_queries",
                    "queries_evaluated", "MAP", "Recall@100", "P@10", "nDCG@10",
                    "elapsed_seconds", "status", "error", "created_at"]
            return [dict(zip(keys, row)) for row in cur.fetchall()]
