from Services.EvaluationService.evaluation_metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# A small hand-built example so we can verify each metric against values computed by hand.
# Ranking returned by the system (best first):
ranked = ["d1", "d2", "d3", "d4", "d5"]
# Graded relevance judgments (qrels) for this query:
graded_relevance = {"d1": 1, "d3": 2, "d5": 1}
relevant = {"d1", "d3", "d5"}


def run_test():
    p3 = precision_at_k(ranked, relevant, k=3)
    r5 = recall_at_k(ranked, relevant, k=5)
    ap = average_precision(ranked, relevant)
    ndcg5 = ndcg_at_k(ranked, graded_relevance, k=5)

    print(f"P@3   = {p3:.4f}   (expected 0.6667)")
    print(f"R@5   = {r5:.4f}   (expected 1.0000)")
    print(f"AP    = {ap:.4f}   (expected 0.7556)")
    print(f"nDCG@5= {ndcg5:.4f}   (expected 0.7624)")


if __name__ == "__main__":
    run_test()
