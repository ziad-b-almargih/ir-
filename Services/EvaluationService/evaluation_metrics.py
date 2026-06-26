import math
from typing import Dict, List, Set


def precision_at_k(ranked_doc_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    # Of the top-k retrieved documents, what fraction are relevant?
    if k <= 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_doc_ids: List[str], relevant: Set[str], k: int = 100) -> float:
    # Of all relevant documents, what fraction did we retrieve within the top-k?
    if not relevant:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def average_precision(ranked_doc_ids: List[str], relevant: Set[str]) -> float:
    # Average of the precision values measured each time a relevant document is hit.
    # Rewards placing relevant documents higher in the ranking.
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for position, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / position
    return precision_sum / len(relevant)


def ndcg_at_k(ranked_doc_ids: List[str], graded_relevance: Dict[str, int], k: int = 10) -> float:
    # DCG: sum of (relevance grade / log2(position + 1)) over the top-k ranking.
    dcg = 0.0
    for position, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        grade = graded_relevance.get(doc_id, 0)
        dcg += grade / math.log2(position + 1)

    # IDCG: the best possible DCG, i.e. the same grades sorted in the ideal order.
    ideal_grades = sorted(graded_relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for position, grade in enumerate(ideal_grades, start=1):
        idcg += grade / math.log2(position + 1)

    return dcg / idcg if idcg > 0 else 0.0
