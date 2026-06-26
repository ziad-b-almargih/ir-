from Services.DataLoaderService.data_loader_service import DataLoaderService
from Services.IndexingService.indexing_service import IndexingService

DATASET_NAME = "beir/quora/test"


def run_test():
    service = IndexingService()
    loader = DataLoaderService(DATASET_NAME)

    # Load everything back FROM DISK (no rebuilding) to prove persistence works.
    corpus = service.load_corpus(DATASET_NAME)
    print(f"[index_test] Loaded corpus with {len(corpus.doc_ids)} documents.")

    query = "How can I invest in the stock market?"

    for strategy_name in ("tfidf", "bm25"):
        strategy = service.load_index(DATASET_NAME, strategy_name)
        scores = strategy.get_scores(query)

        # Rank by score, then fetch only the top documents' text for display.
        ranked_positions = scores.argsort()[::-1][:3]
        top_doc_ids = [corpus.doc_ids[pos] for pos in ranked_positions]
        texts = loader.get_texts(top_doc_ids)

        print("\n" + "=" * 60)
        print(f"[{strategy_name}] Results for: {query}")
        print("=" * 60)
        for rank, pos in enumerate(ranked_positions, start=1):
            doc_id = corpus.doc_ids[pos]
            print(f"Rank {rank} | Score: {scores[pos]:.4f} | {texts[doc_id][:90]}")


if __name__ == "__main__":
    run_test()
