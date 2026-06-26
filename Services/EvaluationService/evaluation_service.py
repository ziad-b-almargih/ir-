from typing import Dict, List, Optional

import numpy as np

from Services.EvaluationService.evaluation_metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from Services.IndexingService.indexing_service import Corpus
from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


class EvaluationService:
    def evaluate(self, strategy: IRepresentationStrategy, corpus: Corpus,
                 queries, qrels: Dict[str, Dict[str, int]],
                 query_limit: Optional[int] = None, cutoff: int = 1000,
                 k_precision: int = 10, k_ndcg: int = 10, k_recall: int = 100,
                 refiner=None) -> Dict[str, float]:

        ap_scores, recall_scores, precision_scores, ndcg_scores = [], [], [], []
        doc_ids = corpus.doc_ids
        evaluated = 0

        for query in queries:
            if query_limit is not None and evaluated >= query_limit:
                break

            # Only evaluate queries that actually have relevance judgments.
            graded_relevance = qrels.get(query.query_id)
            if not graded_relevance:
                continue

            # A document counts as relevant when its grade is greater than zero.
            relevant = {doc_id for doc_id, grade in graded_relevance.items() if grade > 0}
            if not relevant:
                continue

            # Optionally refine the query (spelling + synonyms) before scoring.
            query_text = refiner.refine(query.text).refined if refiner is not None else query.text

            # Each strategy preprocesses the raw query itself, matching how it indexed.
            scores = strategy.get_scores(query_text)
            ranked_doc_ids = self._rank(scores, doc_ids, cutoff)

            ap_scores.append(average_precision(ranked_doc_ids, relevant))
            recall_scores.append(recall_at_k(ranked_doc_ids, relevant, k_recall))
            precision_scores.append(precision_at_k(ranked_doc_ids, relevant, k_precision))
            ndcg_scores.append(ndcg_at_k(ranked_doc_ids, graded_relevance, k_ndcg))
            evaluated += 1

        return {
            "queries_evaluated": evaluated,
            "MAP": self._mean(ap_scores),
            f"Recall@{k_recall}": self._mean(recall_scores),
            f"P@{k_precision}": self._mean(precision_scores),
            f"nDCG@{k_ndcg}": self._mean(ndcg_scores),
        }

    @staticmethod
    def _rank(scores: np.ndarray, doc_ids: List[str], cutoff: int) -> List[str]:
        # Take the top-`cutoff` document positions by score, then map them to document ids.
        scores = np.asarray(scores)
        top_positions = np.argsort(scores)[::-1][:cutoff]
        return [doc_ids[position] for position in top_positions]

    @staticmethod
    def _mean(values: List[float]) -> float:
        return round(float(np.mean(values)), 4) if values else 0.0
