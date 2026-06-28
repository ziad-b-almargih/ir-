from typing import Dict, List, Optional

import numpy as np

from Services.DocumentStoreService.document_store_service import DocumentStoreService
from Services.IndexingService.indexing_service import IndexingService
from Services.IndexingService.persistence import Persistence
from Services.QueryHistoryService.query_history_service import QueryHistoryService
from Services.QueryRefinementService.query_refinement_service import QueryRefinementService
from Services.RegistryService.registry_service import RegistryService
from Services.RepresentationService.hybrid_parallel_strategy import HybridParallelStrategy
from Services.RepresentationService.hybrid_serial_strategy import HybridSerialStrategy
from Services.RepresentationService.hybrid_weighted_strategy import HybridWeightedStrategy
from Services.TopicService.topic_service import TopicService

# Hybrid models are composed on the fly from already-built children, not loaded from disk.
HYBRID_MODELS = {"hybrid_serial", "hybrid_parallel", "hybrid_weighted"}


class SearchService:
    """Orchestrates a query end to end and caches loaded indexes in memory.

    Loading a 30-40 MB index from disk is slow, so each (dataset, model) index is loaded
    once and reused across requests. Raw text for the results is fetched from the database
    by id at query time (never from slow text files).
    """

    def __init__(self):
        self._indexing = IndexingService()
        self._registry = RegistryService()
        self._docstore = DocumentStoreService()
        self._topics = TopicService()
        self._index_cache: Dict[str, object] = {}
        self._corpus_cache: Dict[str, object] = {}
        self._topic_ids_cache: Dict[tuple, set] = {}  # (dataset, topic_id) -> set of doc_ids
        self._qrels_cache: Dict[str, list] = {}  # dataset -> [{"query_id", "text", "doc_ids"}]
        self._refiner: Optional[QueryRefinementService] = None
        self._history: Optional[QueryHistoryService] = None

    @property
    def refiner(self) -> QueryRefinementService:
        # Loaded lazily because building the spell checker takes a moment.
        if self._refiner is None:
            self._refiner = QueryRefinementService()
        return self._refiner

    @property
    def history(self) -> QueryHistoryService:
        # Lazy: encoder load is ~1s and only needed once the user actually searches.
        if self._history is None:
            self._history = QueryHistoryService()
        return self._history

    def test_query(self, dataset_name: str, model: str, query_id: str, top_k: int = 10,
                   k1: Optional[float] = None, b: Optional[float] = None,
                   refine: bool = False, topic_id: Optional[int] = None, alpha: Optional[float] = None) -> Dict:
        # Run the model on a judged test query and report per-query metrics against qrels.
        from Services.EvaluationService.evaluation_metrics import (
            average_precision, ndcg_at_k, precision_at_k, recall_at_k)

        # Look up the qrel entry (cached).
        entry = next((e for e in self._qrels_index(dataset_name)
                      if e["query_id"] == str(query_id)), None)
        if entry is None:
            raise ValueError(f"query_id '{query_id}' not found in qrels for '{dataset_name}'.")

        strategy = self._get_index(dataset_name, model, alpha)
        corpus = self._get_corpus(dataset_name)

        query_used = entry["text"]
        if refine:
            query_used = self.refiner.refine(query_used).refined

        if model == "bm25" and (k1 is not None or b is not None):
            scores = strategy.get_scores_with_params(
                query_used, k1 if k1 is not None else strategy.k1,
                b if b is not None else strategy.b)
        else:
            scores = strategy.get_scores(query_used)

        if topic_id is not None:
            scores = self._mask_by_topic(scores, corpus.doc_ids, dataset_name, topic_id)

        # Rank a wider window so Recall@100 is meaningful even when top_k is small.
        cutoff = max(100, top_k)
        scores = np.asarray(scores)
        ranked_positions = scores.argsort()[::-1][:cutoff]
        ranked_doc_ids = [corpus.doc_ids[pos] for pos in ranked_positions]

        graded_relevance = entry["relevance"]
        relevant = {d for d, g in graded_relevance.items() if g > 0}

        metrics = {
            "AP": round(float(average_precision(ranked_doc_ids, relevant)), 4),
            "Recall@100": round(float(recall_at_k(ranked_doc_ids, relevant, 100)), 4),
            "P@10": round(float(precision_at_k(ranked_doc_ids, relevant, 10)), 4),
            "nDCG@10": round(float(ndcg_at_k(ranked_doc_ids, graded_relevance, 10)), 4),
            "relevant_in_qrels": len(relevant),
        }

        # Top-K for the UI, fetched from the database, annotated with relevance info.
        top_doc_ids = ranked_doc_ids[:top_k]
        texts = self._docstore.get_texts(dataset_name, top_doc_ids)
        results = []
        for rank, position in enumerate(ranked_positions[:top_k], start=1):
            doc_id = corpus.doc_ids[position]
            score = float(scores[position])
            if not np.isfinite(score) or score <= 0:
                continue
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(score, 4),
                "text": texts.get(doc_id, ""),
                "relevant": doc_id in relevant,
                "grade": int(graded_relevance.get(doc_id, 0)),
            })

        return {
            "query_id": entry["query_id"],
            "query_text": entry["text"],
            "query_used": query_used,
            "metrics": metrics,
            "results": results,
        }

    def get_documents(self, dataset_name: str, doc_ids: List[str]) -> Dict[str, str]:
        # Thin pass-through so the API has one consistent place to fetch raw texts by id.
        return self._docstore.get_texts(dataset_name, doc_ids)

    def browse_qrels(self, dataset_name: str, query: str = "",
                     limit: int = 50, offset: int = 0) -> Dict:
        # Returns judged test queries with their relevant doc_ids, paginated.
        all_entries = self._qrels_index(dataset_name)
        if query:
            needle = query.lower()
            filtered = [e for e in all_entries
                        if needle in e["text"].lower() or needle == e["query_id"]]
        else:
            filtered = all_entries
        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "queries": filtered[offset:offset + limit],
        }

    def _qrels_index(self, dataset_name: str) -> list:
        # Heavy first call (~3s on Quora); cached afterwards.
        if dataset_name in self._qrels_cache:
            return self._qrels_cache[dataset_name]
        from Services.DataLoaderService.data_loader_service import DataLoaderService
        loader = DataLoaderService(dataset_name)
        qrels = loader.load_qrels()
        entries = []
        for query in loader.load_queries():
            relevance = qrels.get(query.query_id)
            if not relevance:
                continue
            # Sort relevant docs by relevance grade (desc), then by id for stability.
            sorted_docs = sorted(relevance.items(), key=lambda kv: (-kv[1], kv[0]))
            entries.append({
                "query_id": query.query_id,
                "text": query.text,
                "doc_ids": [doc_id for doc_id, _ in sorted_docs],
                "relevance": {doc_id: grade for doc_id, grade in sorted_docs},
            })
        self._qrels_cache[dataset_name] = entries
        return entries

    def suggest(self, query: str, max_suggestions: int = 6) -> List[str]:
        # Alternative query phrasings: refiner rewrites + nearest past queries from history.
        suggestions = list(self.refiner.suggest(query, max_suggestions))
        seen = {query.strip().lower(), *(s.lower() for s in suggestions)}
        try:
            for text, _sim in self.history.similar_queries(query, k=max_suggestions):
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(text)
                if len(suggestions) >= max_suggestions:
                    break
        except Exception as error:  # noqa: BLE001 — history must never break suggest()
            print(f"[SearchService] history suggest skipped: {error}")
        return suggestions[:max_suggestions]

    def warmup(self) -> None:
        # Preload corpus + indexes for every ready dataset so the first query is fast.
        self._docstore.ensure_schema()
        for dataset in self.available_datasets():
            try:
                self._get_corpus(dataset)
                for model in self.available_models(dataset):
                    self._get_index(dataset, model)
            except Exception as error:  # noqa: BLE001
                print(f"[SearchService] Warmup skipped {dataset}: {error}")

    def available_datasets(self) -> List[str]:
        return self._registry.ready_datasets()

    def available_models(self, dataset_name: str) -> List[str]:
        entry = self._registry.get(dataset_name)
        if not entry:
            return []
        # Only offer models whose index file is actually on disk.
        models = [m for m in entry.get("models", []) if Persistence.exists(f"{dataset_name}__{m}")]
        # Expose hybrids automatically when both BM25 and Embedding are ready.
        if "bm25" in models and "embedding" in models:
            models.extend(["hybrid_serial", "hybrid_parallel", "hybrid_weighted"])
        return models

    def search(self, dataset_name: str, model: str, query: str, top_k: int = 10,
               k1: Optional[float] = None, b: Optional[float] = None,
               refine: bool = False, use_history: bool = False,
               alpha: Optional[float] = None,
               topic_id: Optional[int] = None) -> Dict:
        strategy = self._get_index(dataset_name, model, alpha)
        corpus = self._get_corpus(dataset_name)

        # Optional query refinement (spelling + synonyms) before scoring.
        query_used = query
        history_expansion: List[str] = []
        if refine:
            query_used = self.refiner.refine(query).refined
        # Independent of refine: silently append terms from semantically similar past queries.
        if use_history:
            try:
                history_expansion = self.history.expansion_terms(query_used)
            except Exception as error:  # noqa: BLE001 — history is best-effort
                print(f"[SearchService] history expansion skipped: {error}")
            if history_expansion:
                query_used = f"{query_used} {' '.join(history_expansion)}"

        # Custom BM25 parameters re-score without rebuilding the index.
        if model == "bm25" and (k1 is not None or b is not None):
            scores = strategy.get_scores_with_params(
                query_used, k1 if k1 is not None else strategy.k1,
                b if b is not None else strategy.b)
        else:
            scores = strategy.get_scores(query_used)

        # Optional topic filter: drop any doc that is not in the chosen topic.
        if topic_id is not None:
            scores = self._mask_by_topic(scores, corpus.doc_ids, dataset_name, topic_id)

        ranked_positions = np.asarray(scores).argsort()[::-1][:top_k]
        top_doc_ids = [corpus.doc_ids[pos] for pos in ranked_positions]
        # Fetch the ORIGINAL raw text of the top results from the database by id.
        texts = self._docstore.get_texts(dataset_name, top_doc_ids)

        results = []
        for rank, pos in enumerate(ranked_positions, start=1):
            score = float(scores[pos])
            if score <= 0:
                continue
            doc_id = corpus.doc_ids[pos]
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(score, 4),
                "text": texts[doc_id],
            })
        # Record the user's ORIGINAL query so future searches can learn from it.
        try:
            self.history.record(query)
        except Exception as error:  # noqa: BLE001 — recording must never break search
            print(f"[SearchService] history record skipped: {error}")

        return {"results": results, "query_used": query_used,
                "history_expansion": history_expansion}

    def _get_index(self, dataset_name: str, model: str, alpha: Optional[float] = None):
        key = f"{dataset_name}__{model}"
        if key in self._index_cache:
            cached = self._index_cache[key]
            # alpha is a per-request knob — update it on the cached instance.
            if model == "hybrid_weighted" and alpha is not None:
                cached.alpha = max(0.0, min(1.0, float(alpha)))
            return cached
        print("[SearchService] Loading index..." + dataset_name + " : " + model)
        if model in HYBRID_MODELS:
            # Compose of the (cached) children — no separate file on disk.
            bm25 = self._get_index(dataset_name, "bm25")
            embedding = self._get_index(dataset_name, "embedding")
            if model == "hybrid_serial":
                self._index_cache[key] = HybridSerialStrategy(bm25, embedding)
            elif model == "hybrid_weighted":
                # Default to 0.5 if no alpha was passed at first build.
                self._index_cache[key] = HybridWeightedStrategy(
                    bm25, embedding, alpha if alpha is not None else 0.5)
            else:  # hybrid_parallel
                self._index_cache[key] = HybridParallelStrategy([bm25, embedding])
            return self._index_cache[key]

        if not Persistence.exists(key):
            raise ValueError(f"Index '{model}' for dataset '{dataset_name}' is not built.")
        self._index_cache[key] = self._indexing.load_index(dataset_name, model)
        return self._index_cache[key]

    def _get_corpus(self, dataset_name: str):
        if dataset_name not in self._corpus_cache:
            self._corpus_cache[dataset_name] = self._indexing.load_corpus(dataset_name)
        return self._corpus_cache[dataset_name]

    def _mask_by_topic(self, scores: np.ndarray, doc_ids, dataset_name: str,
                       topic_id: int) -> np.ndarray:
        key = (dataset_name, topic_id)
        if key not in self._topic_ids_cache:
            self._topic_ids_cache[key] = set(self._topics.doc_ids_for_topic(dataset_name, topic_id))
        topic_doc_ids = self._topic_ids_cache[key]
        if not topic_doc_ids:
            return np.full_like(scores, -np.inf, dtype=np.float64)
        mask = np.fromiter((doc_id in topic_doc_ids for doc_id in doc_ids), dtype=bool,
                           count=len(doc_ids))
        return np.where(mask, scores, -np.inf)
