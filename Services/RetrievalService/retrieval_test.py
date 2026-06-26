from Services.RepresentationService.bm25_strategy import Bm25Strategy
from Services.RepresentationService.tfidf_strategy import TfIdfStrategy
from Services.RepresentationService.representation_context import RepresentationContext
from Services.RetrievalService.retrieval_service import RetrievalService


def run_test():
    raw_docs = [
        "The system architecture handles complex SQL queries perfectly.",
        "Optimizing complex SQL queries improves the database architecture.",
        "A good database architecture handles both simple and complex queries.",
        "Information Retrieval systems rely on TF-IDF for document ranking.",
        "The new search engine uses BM25 and TF-IDF for better document ranking.",
        "Advanced document ranking relies on BM25 rather than just TF-IDF models.",
        "Building microservices ensures scalability and clean system architecture.",
        "An API gateway connects microservices to improve the overall architecture."
    ]

    query = "database system"

    # Both strategies are consumed through the exact same code path now.
    strategies = {
        "TF-IDF": TfIdfStrategy(),
        "BM25": Bm25Strategy(),
    }

    for name, strategy in strategies.items():
        context = RepresentationContext(strategy)
        context.index_documents(raw_docs)
        scores = context.get_scores(query)
        results = RetrievalService.get_top_results(scores, raw_docs, top_k=3)

        print("\n" + "=" * 50)
        print(f"[{name}] Search results for: {query}")
        print("=" * 50)
        if not results:
            print("No matching documents found.")
        else:
            for res in results:
                print(f"Rank {res['rank']} | Score: {res['score']} | Doc: {res['document']}")


if __name__ == "__main__":
    run_test()
