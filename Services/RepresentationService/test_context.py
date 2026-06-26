from Services.RepresentationService.tfidf_strategy import TfIdfStrategy
from Services.RepresentationService.representation_context import RepresentationContext


def run_test():
    context = RepresentationContext(TfIdfStrategy())

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

    context.index_documents(raw_docs)

    scores = context.get_scores("database")
    print("-> Scores shape (one score per document):", scores.shape)
    print("-> Scores:", scores.round(3))


if __name__ == "__main__":
    run_test()
