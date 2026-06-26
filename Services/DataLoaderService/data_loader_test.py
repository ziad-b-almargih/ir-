from Services.DataLoaderService.data_loader_service import DataLoaderService

# Quora is the lighter of our two collections, so we use it for a quick smoke test.
DATASET_NAME = "beir/quora/test"


def run_test():
    loader = DataLoaderService(DATASET_NAME)

    print(f"[DataLoader] Dataset: {DATASET_NAME}")
    print(f"[DataLoader] Total documents: {loader.doc_count()}")

    sample_docs = loader.load_documents(limit=3)
    print("\n--- Sample documents ---")
    for doc in sample_docs:
        print(f"[{doc.doc_id}] {doc.text[:100]}")

    queries = loader.load_queries()
    print(f"\n--- Queries: {len(queries)} total ---")
    for query in queries[:3]:
        print(f"[{query.query_id}] {query.text}")

    qrels = loader.load_qrels()
    print(f"\n--- Qrels: {len(qrels)} queries have judgments ---")
    sample_qid = next(iter(qrels))
    print(f"Example qrels for query {sample_qid}: {qrels[sample_qid]}")


if __name__ == "__main__":
    run_test()
