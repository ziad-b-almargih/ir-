import random
import time
from typing import Dict, Optional

from Services.DataLoaderService.data_loader_service import DataLoaderService
from Services.IndexingService.indexing_service import IndexingService
from Services.EvaluationService.evaluation_service import EvaluationService


def evaluate_dataset(dataset_name: str, model: str,
                     num_queries: Optional[int] = None,
                     refine: bool = False,
                     random_sample: bool = False, seed: int = 42) -> Dict:
    """Run evaluation for one (dataset, model).

    Per professor's instruction, the default is to use ALL judged queries (no sampling).
    `num_queries` is only there to allow a quick partial run when exploring.
    """
    # Resolve dependencies — heavy I/O happens here, before the evaluation loop.
    from Services.SearchService.search_service import SearchService
    search = SearchService()
    indexing = IndexingService()
    loader = DataLoaderService(dataset_name)
    evaluator = EvaluationService()

    corpus = indexing.load_corpus(dataset_name)
    # Use SearchService's lookup so hybrid models compose automatically from their children.
    strategy = search._get_index(dataset_name, model)
    qrels = loader.load_qrels()

    judged_queries = [q for q in loader.load_queries() if q.query_id in qrels]
    total_judged = len(judged_queries)

    if num_queries is None or num_queries >= total_judged:
        sample = judged_queries
        used_all = True
    else:
        if random_sample:
            random.Random(seed).shuffle(judged_queries)
        sample = judged_queries[:num_queries]
        used_all = False

    n = len(sample)
    # Loud, notebook-friendly logging — graders want the count visible in the output.
    print("=" * 60)
    print(f"▶ Evaluating  model='{model}'  dataset='{dataset_name}'  refine={refine}")
    print(f"   judged queries in qrels: {total_judged}")
    print(f"   queries used in this run: N = {n}  (used_all_qrels={used_all})")
    print("=" * 60)

    # Build a refiner only when refinement is requested (it's heavy to construct).
    refiner = None
    if refine:
        from Services.QueryRefinementService.query_refinement_service import QueryRefinementService
        refiner = QueryRefinementService()

    started = time.time()
    metrics = evaluator.evaluate(strategy, corpus, sample, qrels, query_limit=n, refiner=refiner)
    elapsed = time.time() - started

    metrics["dataset"] = dataset_name
    metrics["model"] = model
    metrics["refine"] = refine
    metrics["judged_queries_in_qrels"] = total_judged
    metrics["queries_used"] = n
    metrics["used_all_qrels"] = used_all
    metrics["elapsed_seconds"] = round(elapsed, 2)

    print(f"✔ Evaluated {metrics['queries_evaluated']} queries in {metrics['elapsed_seconds']}s.")
    print(f"   MAP={metrics.get('MAP')}  nDCG@10={metrics.get('nDCG@10')}  "
          f"P@10={metrics.get('P@10')}  Recall@100={metrics.get('Recall@100')}")
    return metrics


def run(dataset_name: str, num_queries: Optional[int] = None):
    for model in ("tfidf", "bm25"):
        print(evaluate_dataset(dataset_name, model, num_queries=num_queries))


if __name__ == "__main__":
    # Default: ALL qrels (this is what the report/notebook should use).
    run("beir/quora/test", num_queries=None)
