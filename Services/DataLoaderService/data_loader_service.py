from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional
import ir_datasets


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Query:
    query_id: str
    text: str


class DataLoaderService:
    """Loads documents, queries and qrels for a single ir_datasets collection.

    qrels are returned as {query_id: {doc_id: relevance}}, the standard shape
    expected by the evaluation metrics (MAP, nDCG, Recall, P@10).
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        # load() is lazy: it only resolves a handle here. Data is downloaded (and cached)
        # the first time we actually iterate documents/queries/qrels.
        self._dataset = ir_datasets.load(dataset_name)

    def iter_documents(self) -> Iterator[Document]:
        # Stream documents one at a time to avoid holding 200K+ docs in memory at once.
        for doc in self._dataset.docs_iter():
            yield Document(doc_id=doc.doc_id, text=self._extract_doc_text(doc))

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        # Materialize documents into a list; limit is mainly for quick local testing.
        documents = []
        for i, document in enumerate(self.iter_documents()):
            if limit is not None and i >= limit:
                break
            documents.append(document)
        return documents

    def load_queries(self) -> List[Query]:
        return [Query(query_id=q.query_id, text=q.text) for q in self._dataset.queries_iter()]

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        qrels: Dict[str, Dict[str, int]] = {}
        for qrel in self._dataset.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        return qrels

    def doc_count(self) -> int:
        # Number of documents in the collection (without iterating the whole corpus).
        return self._dataset.docs_count()

    def get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        # Random-access lookup for a few documents (e.g. the top results to display),
        # without loading the whole corpus into memory.
        store = self._dataset.docs_store()
        texts = {}
        for doc_id in doc_ids:
            texts[doc_id] = self._extract_doc_text(store.get(doc_id))
        return texts

    @staticmethod
    def _extract_doc_text(doc) -> str:
        # BEIR documents typically carry a title plus a body; merge them when both exist.
        title = getattr(doc, "title", "") or ""
        body = getattr(doc, "text", "") or ""
        return f"{title} {body}".strip()
