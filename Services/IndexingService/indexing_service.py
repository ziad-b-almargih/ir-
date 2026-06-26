from dataclasses import dataclass
from typing import Iterator, List, Optional

from Services.DataLoaderService.data_loader_service import DataLoaderService
from Services.IndexingService.persistence import Persistence
from Services.RepresentationService.i_representation_strategy import IRepresentationStrategy


@dataclass
class Corpus:
    # Lightweight handle: only the ordered id list lives here. Row i of every index
    # corresponds to doc_ids[i]. Raw texts are fetched lazily from ir_datasets.
    doc_ids: List[str]


class IndexingService:
    def build_corpus(self, dataset_name: str, limit: Optional[int] = None) -> Corpus:
        # Just record the document ids in collection order (cheap: no preprocessing here).
        loader = DataLoaderService(dataset_name)
        doc_ids: List[str] = []
        for i, document in enumerate(loader.iter_documents()):
            if limit is not None and i >= limit:
                break
            doc_ids.append(document.doc_id)

        corpus = Corpus(doc_ids=doc_ids)
        Persistence.save(corpus, self._corpus_key(dataset_name))
        print(f"[IndexingService] Corpus saved: {len(doc_ids)} document ids.")
        return corpus

    def load_corpus(self, dataset_name: str) -> Corpus:
        return Persistence.load(self._corpus_key(dataset_name))

    def iter_raw_texts(self, dataset_name: str, limit: Optional[int] = None) -> Iterator[str]:
        # Stream raw document texts in the same order as doc_ids. Each strategy applies
        # whatever preprocessing it needs (lexical models tokenize, embeddings do not).
        loader = DataLoaderService(dataset_name)
        for i, document in enumerate(loader.iter_documents()):
            if limit is not None and i >= limit:
                break
            yield document.text

    def build_index(self, dataset_name: str, strategy_name: str,
                    strategy: IRepresentationStrategy) -> IRepresentationStrategy:
        # Every strategy receives the same raw-text stream and owns its own preprocessing.
        strategy.index_documents(self.iter_raw_texts(dataset_name))
        Persistence.save(strategy, self._index_key(dataset_name, strategy_name))
        print(f"[IndexingService] Index '{strategy_name}' built and saved.")
        return strategy

    def load_index(self, dataset_name: str, strategy_name: str) -> IRepresentationStrategy:
        return Persistence.load(self._index_key(dataset_name, strategy_name))

    @staticmethod
    def _corpus_key(dataset_name: str) -> str:
        return f"{dataset_name}__corpus"

    @staticmethod
    def _index_key(dataset_name: str, strategy_name: str) -> str:
        return f"{dataset_name}__{strategy_name}"
