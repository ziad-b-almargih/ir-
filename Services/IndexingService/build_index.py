import sys

from Services.DocumentStoreService.document_store_service import DocumentStoreService
from Services.IndexingService.indexing_service import IndexingService
from Services.RepresentationService.tfidf_strategy import TfIdfStrategy
from Services.RepresentationService.bm25_strategy import Bm25Strategy
from Services.RepresentationService.embedding_strategy import EmbeddingStrategy


def build_corpus(dataset_name: str, limit=None):
    IndexingService().build_corpus(dataset_name, limit=limit)


def load_db(dataset_name: str):
    DocumentStoreService().load_dataset(dataset_name)


def build_tfidf(dataset_name: str):
    IndexingService().build_index(dataset_name, "tfidf", TfIdfStrategy())


def build_bm25(dataset_name: str):
    IndexingService().build_index(dataset_name, "bm25", Bm25Strategy())


def build_embedding(dataset_name: str):
    IndexingService().build_index(dataset_name, "embedding", EmbeddingStrategy())



# Maps a model name to a fresh strategy instance.
STRATEGY_FACTORY = {
    "tfidf": TfIdfStrategy,
    "bm25": Bm25Strategy,
    "embedding": EmbeddingStrategy,
}


def build_database(dataset_name: str, models: list) -> int:
    # Offline pipeline: corpus ids, raw text into the DB, then each representation.
    service = IndexingService()
    corpus = service.build_corpus(dataset_name)
    DocumentStoreService().load_dataset(dataset_name)
    for model in models:
        if model not in STRATEGY_FACTORY:
            raise ValueError(f"Unknown model '{model}'.")
        service.build_index(dataset_name, model, STRATEGY_FACTORY[model]())
    return len(corpus.doc_ids)


# Each step is a separate entry point so it can run in its own process and release
# all memory before the next step starts (important for large collections).
STEPS = {"corpus": build_corpus, "db": load_db, "tfidf": build_tfidf,
         "bm25": build_bm25, "embedding": build_embedding}


if __name__ == "__main__":
    # Usage: python -m Services.IndexingService.build_index <step> <dataset_name>
    step = sys.argv[1]
    dataset = sys.argv[2]
    STEPS[step](dataset)
    print(f"[build_index] Step '{step}' done for {dataset}.")
