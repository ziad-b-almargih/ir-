import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Services.EvaluationService.run_evaluation import evaluate_dataset
from Services.EvaluationStoreService.evaluation_store_service import EvaluationStoreService
from Services.IndexingService.build_index import STRATEGY_FACTORY, build_database
from Services.RegistryService.registry_service import (
    RegistryService, STATUS_BUILDING, STATUS_ERROR, STATUS_READY)
from Services.SearchService.search_service import SearchService
from Services.TopicService.topic_service import TopicService

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

# One shared SearchService keeps the loaded indexes cached across requests.
search_service = SearchService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the index cache at startup so the first user query is fast (< 20s budget).
    search_service.warmup()
    yield


app = FastAPI(title="IR Search Engine API", version="2.0", lifespan=lifespan)
api = APIRouter(prefix="/api")


# ----------------------------- request/response models -----------------------------
class SearchRequest(BaseModel):
    dataset: str
    model: str
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    k1: Optional[float] = None
    b: Optional[float] = None
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)  # hybrid_weighted: lex vs sem balance
    refine: bool = False
    use_history: bool = False        # silently expand query using semantically similar past queries
    topic_id: Optional[int] = None   # if set, restrict results to docs in this topic


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    dataset: str
    model: str
    query: str
    query_used: str
    history_expansion: List[str] = []
    count: int
    results: List[SearchResult]


class DatabaseRequest(BaseModel):
    dataset: str = Field(..., min_length=1, description="ir_datasets id, e.g. 'beir/quora/test'")
    models: List[str] = Field(..., min_length=1)


class EvaluateRequest(BaseModel):
    dataset: str
    model: str
    num_queries: Optional[int] = Field(50, ge=1, le=50000)
    all_queries: bool = False     # if true, evaluate on ALL judged queries (overrides num_queries)
    refine: bool = False
    save: bool = False             # forced True when all_queries is True


class SuggestRequest(BaseModel):
    query: str = Field(..., min_length=1)


class TestQueryRequest(BaseModel):
    dataset: str
    model: str
    query_id: str
    top_k: int = Field(10, ge=1, le=100)
    k1: Optional[float] = None
    b: Optional[float] = None
    refine: bool = False
    topic_id: Optional[int] = None
    alpha: Optional[float] = None


class BuildTopicRequest(BaseModel):
    dataset: str = Field(..., min_length=1, description="ir_datasets id, e.g. 'beir/quora/test'")

# ----------------------------- background build task -----------------------------
def _build_database_task(dataset: str, models: List[str]):
    registry = RegistryService()
    try:
        doc_count = build_database(dataset, models)
        registry.upsert(dataset, models=models, status=STATUS_READY, doc_count=doc_count)
    except Exception as error:  # noqa: BLE001 - surface any build failure in the registry
        registry.upsert(dataset, status=STATUS_ERROR, error=str(error))


# ----------------------------- API routes -----------------------------
@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/datasets")
def datasets():
    return {"datasets": search_service.available_datasets()}


@api.get("/datasets/{dataset:path}/models")
def models(dataset: str):
    return {"dataset": dataset, "models": search_service.available_models(dataset)}


@api.get("/databases")
def list_databases():
    registry = RegistryService().list_databases()
    return {"databases": [{"dataset": name, **info} for name, info in registry.items()]}


@api.post("/databases")
def add_database(request: DatabaseRequest, background: BackgroundTasks):
    unknown = [m for m in request.models if m not in STRATEGY_FACTORY]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown models: {unknown}")

    # Mark as building immediately, then build in the background.
    RegistryService().upsert(request.dataset, models=request.models, status=STATUS_BUILDING)
    background.add_task(_build_database_task, request.dataset, request.models)
    return {"dataset": request.dataset, "status": STATUS_BUILDING}


def _run_full_evaluation_bg(run_id: int, dataset: str, model: str, refine: bool):
    store = EvaluationStoreService()
    try:
        metrics = evaluate_dataset(dataset, model, num_queries=None, refine=refine)
        store.finalize(run_id, metrics)
    except Exception as error:  # noqa: BLE001
        store.fail(run_id, str(error))


@api.post("/evaluate")
def evaluate(request: EvaluateRequest, background: BackgroundTasks):
    # All-queries runs persist automatically and run in the background (they can take hours).
    if request.all_queries:
        store = EvaluationStoreService()
        run_id = store.create_running(request.dataset, request.model, request.refine,
                                      used_all_qrels=True, num_queries=0)
        background.add_task(_run_full_evaluation_bg, run_id, request.dataset,
                            request.model, request.refine)
        return {"status": "queued", "run_id": run_id,
                "message": "Full-qrels evaluation started in the background."}

    # Partial (sample) runs are synchronous; persistence is optional.
    try:
        metrics = evaluate_dataset(request.dataset, request.model,
                                   num_queries=request.num_queries, refine=request.refine)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(error))

    if request.save:
        EvaluationStoreService().save_completed(request.dataset, request.model,
                                                request.refine, used_all_qrels=False,
                                                metrics=metrics)
    return metrics


@api.get("/evaluations")
def evaluations(dataset: Optional[str] = None):
    return {"evaluations": EvaluationStoreService().list(dataset)}


@api.get("/topics")
def topics(dataset: str):
    # Returns one row per topic with its top words and assigned document count.
    return {"dataset": dataset, "topics": TopicService().list_topics(dataset)}


@api.get("/qrels")
def qrels(dataset: str, q: str = "", limit: int = 50, offset: int = 0):
    # Paginated browser over the dataset's test queries with their relevant doc_ids.
    return search_service.browse_qrels(dataset, q, limit, offset)


@api.get("/documents")
def documents(dataset: str, ids: str):
    # Bulk fetch raw text for a comma-separated list of doc_ids.
    doc_ids = [i for i in ids.split(",") if i]
    texts = search_service.get_documents(dataset, doc_ids)
    return {"documents": [{"doc_id": d, "text": texts.get(d, "")} for d in doc_ids]}

@api.post("/build-topic")
def suggest(request: BuildTopicRequest):
    topic_service = TopicService()
    topic_service.build(request.dataset)
    return {"dataset": request.dataset, "topics": topic_service.list_topics(request.dataset)}



@api.post("/suggest")
def suggest(request: SuggestRequest):
    return {"query": request.query, "suggestions": search_service.suggest(request.query)}


@api.post("/test-query")
def test_query(request: TestQueryRequest):
    # Same retrieval pipeline as /search, but the input is a judged qrel and the response
    # also contains per-query metrics (AP, P@10, nDCG@10, Recall@100).
    try:
        return search_service.test_query(
            dataset_name=request.dataset, model=request.model, query_id=request.query_id,
            top_k=request.top_k, k1=request.k1, b=request.b,
            refine=request.refine, topic_id=request.topic_id, alpha = request.alpha)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@api.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        outcome = search_service.search(
            dataset_name=request.dataset, model=request.model, query=request.query,
            top_k=request.top_k, k1=request.k1, b=request.b, refine=request.refine,
            use_history=request.use_history, alpha=request.alpha,
            topic_id=request.topic_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    results = outcome["results"]
    return SearchResponse(
        dataset=request.dataset, model=request.model, query=request.query,
        query_used=outcome["query_used"],
        history_expansion=outcome.get("history_expansion", []),
        count=len(results), results=results)



app.include_router(api)
# Serve the dark-theme frontend at the root (mounted last so /api/* wins).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
