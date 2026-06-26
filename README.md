# IR Search Engine

Custom Information Retrieval system built with a Service-Oriented Architecture (SOA).
It indexes large document collections with several representation models, fetches the
**original raw text from PostgreSQL by document id** at query time, exposes a REST API
(FastAPI), and ships a dark-theme web UI for searching, evaluating, and exploring
topics.

---

## 1. Features

**Representations (5 models)**
- TF-IDF (sklearn)
- BM25 (rank_bm25) with **per-query `k1`/`b` tuning** (no re-build needed)
- Embedding (sentence-transformers `all-MiniLM-L6-v2`)
- Hybrid Serial (BM25 shortlists → Embedding re-ranks)
- Hybrid Parallel (Reciprocal Rank Fusion of BM25 + Embedding)

**Storage**
- PostgreSQL `documents` table — raw text fetched by id at query time.
- PostgreSQL `databases` table — registry of built datasets and their models.
- PostgreSQL `evaluation_runs` table — every evaluation run with metrics, refine flag, elapsed time.
- PostgreSQL `topics` + `doc_topics` tables — topical clusters and per-document assignments.
- `data_store/*.joblib` — compact fitted indexes loaded fully into memory once at startup.

**Query refinement (suggestions)**
- Spelling correction (`pyspellchecker`) and WordNet synonym expansion.
- Returned as **clickable suggestion chips** the user picks from (not auto-replace).

**Evaluation**
- MAP, Recall@100, P@10, nDCG@10.
- Toggle **"Run on ALL qrels"** (default per professor's rule); partial sample available for quick checks.
- Runs persisted in Postgres with elapsed time, refine flag, and used_all_qrels flag.
- Bar charts (MAP and nDCG@10) per model, raw vs refined.

**Extra feature: BERTopic topic modeling**
- Topical clusters discovered from the existing embeddings (no re-encoding).
- Search can be **filtered by topic**.
- "Topics" view: table of topics with top words + bar chart of cluster sizes.

**Architecture**
- Strict SOA: each concern is an isolated service under `Services/`.
- API gateway (`api_gateway/`) exposes everything under `/api/...`.
- Frontend (`frontend/`) is plain HTML/CSS/JS served by the same FastAPI app.
- Startup **warmup** preloads all indexes so the first user query is fast (the 20 s budget is met).

---

## 2. Architecture

```
                            ┌──────────────────────┐
                            │   Browser  (UI)      │
                            └──────────┬───────────┘
                                       │  /api/...
                            ┌──────────▼───────────┐
                            │   API Gateway        │  FastAPI
                            └──────────┬───────────┘
       ┌──────────┬──────────┬─────────┼─────────┬───────────────┐
       │          │          │         │         │               │
 SearchService  Indexing  Preproc.  QueryRefine  EvaluationStore TopicService
       │          │                                        ▲
       │          ▼                                        │
       │   data_store/*.joblib                  evaluation_runs (Postgres)
       │     (TF-IDF, BM25,                           ▲
       │      Embedding indexes)                      │
       │                                          EvaluationService
       │                                              ▲
       ▼                                              │
 DocumentStoreService ───────────► PostgreSQL ──── RegistryService
   (raw text by id)                  ir_db        (databases table)
                                      │
                          ┌───────────┼──────────────┐
                       documents   databases     evaluation_runs
                       doc_topics   topics
```

### Services (`Services/`)

| Service | Responsibility |
|---|---|
| `DataLoaderService` | Stream docs/queries/qrels from `ir_datasets` |
| `PreProcessingService` | Tokenize, lowercase, remove punctuation/stopwords, stem (Decorator pattern) |
| `RepresentationService` | TF-IDF, BM25, Embedding, Hybrid (Strategy pattern; unified `get_scores(raw_query)`) |
| `IndexingService` | Build & persist corpus + indexes; load them |
| `RetrievalService` | Pure ranking from a uniform score array |
| `SearchService` | Orchestrate a query end-to-end; cache loaded indexes; optional topic filter |
| `QueryRefinementService` | Spelling correction + WordNet synonyms; clickable suggestions |
| `EvaluationService` | MAP, Recall@100, P@10, nDCG@10 |
| `EvaluationStoreService` | Persist evaluation runs in Postgres |
| `RegistryService` | Track which datasets are built (Postgres `databases` table) |
| `DocumentStoreService` | Bulk-load raw text into Postgres; fetch by id at query time |
| `TopicService` | Fit BERTopic on existing embeddings; persist topics & doc→topic mapping |

---

## 3. Offline vs Online phases

**Offline (preparation, files allowed):**
1. Stream the dataset from `ir_datasets`.
2. Preprocess and build representations (TF-IDF / BM25 / Embedding) → `data_store/*.joblib`.
3. Load raw documents into Postgres `documents` (keyed by `doc_id`).
4. Optionally fit BERTopic → `topics` + `doc_topics`.

**Online (user query, no training, ≤ 20 s):**
1. Strategy preprocesses the raw query internally and scores all documents.
2. Optional topic filter masks docs outside the chosen cluster.
3. Top-10 ids returned.
4. **Original raw text fetched from Postgres by id** (never from text files).

---

## 4. Quick start with Docker (recommended)

Requires Docker and Docker Compose.

```bash
# 1) Build and start: Postgres + the IR app
docker compose up -d --build

# 2) Watch logs the first time (downloads NLTK at build time, then starts uvicorn)
docker compose logs -f app

# 3) Open the UI
# →  http://localhost:8000
# API docs: http://localhost:8000/docs
```

The first build downloads heavy ML wheels (torch, sentence-transformers) and can take
several minutes. Subsequent starts are fast because `data_store/`, the dataset cache,
HuggingFace cache, and Postgres data are all kept in named volumes.

### Build a dataset (from inside the container)

A fresh deployment has no built indexes. Build them via the **Settings** page in the UI,
or from the command line:

```bash
# Run inside the app container — builds corpus + DB load + all selected indexes
docker compose exec app python -m Services.IndexingService.build_index corpus    beir/quora/test
docker compose exec app python -m Services.IndexingService.build_index db        beir/quora/test
docker compose exec app python -m Services.IndexingService.build_index tfidf     beir/quora/test
docker compose exec app python -m Services.IndexingService.build_index bm25      beir/quora/test
docker compose exec app python -m Services.IndexingService.build_index embedding beir/quora/test   # ~30 min on CPU
docker compose exec app python -m Services.IndexingService.build_index topics    beir/quora/test   # after embedding

# Register the built models in the registry so the UI shows them
docker compose exec app python -c "
from Services.RegistryService.registry_service import RegistryService, STATUS_READY
from Services.IndexingService.indexing_service import IndexingService
c = IndexingService().load_corpus('beir/quora/test')
RegistryService().upsert('beir/quora/test', models=['tfidf','bm25','embedding'],
                          status=STATUS_READY, doc_count=len(c.doc_ids))
"
```

After that, restart the app once so warmup loads the new indexes:
```bash
docker compose restart app
```

### Tearing down

```bash
docker compose down            # stop containers, keep data
docker compose down -v         # also wipe Postgres + caches (full reset)
```

---

## 5. Manual setup (without Docker)

Requires Python 3.12 and a local PostgreSQL with a `ir_db` database.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Postgres connection (read by every service). Override any of these as needed.
export PGHOST=localhost PGPORT=5432 PGDATABASE=ir_db PGUSER=postgres PGPASSWORD=12345678

# Build a dataset (same commands as above, without `docker compose exec app`)
python -m Services.IndexingService.build_index corpus beir/quora/test
# ... etc.

# Run the API + UI
uvicorn api_gateway.main:app --reload
# →  http://127.0.0.1:8000
```

---

## 6. Using the UI

The app has four pages (top-right nav):

- **Search**: pick a database, model (TF-IDF/BM25/Embedding/Hybrid Serial/Hybrid Parallel),
  set `top_k`, optionally tune BM25's `k1`/`b`, optionally filter by topic, click **Suggest**
  for alternative query phrasings, run. Each result shows **doc_id**, score, and the
  **original raw text** fetched from Postgres.

- **Evaluation**: pick model + dataset, choose a sample size or **"Run on ALL qrels"**
  (auto-saves to DB), optionally apply refinement, hit **Run**. Saved runs appear in a table
  with elapsed time. Two charts compare MAP and nDCG@10 across models (raw vs refined).

- **Topics**: browse the BERTopic clusters with their top words and sizes; a bar chart shows
  the top 20 topics by size.

- **Settings (⚙)**: add a new database (enter an `ir_datasets` id and pick models). The
  build runs in the background; the databases table shows its status (`building` / `ready` /
  `error`).

---

## 7. API reference (under `/api`)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | Liveness probe |
| GET  | `/api/datasets` | List ready datasets |
| GET  | `/api/datasets/{dataset:path}/models` | List models available for a dataset |
| GET  | `/api/databases` | All registry entries (any status) |
| POST | `/api/databases` | Start building a new database in the background |
| POST | `/api/search` | Search a dataset (supports `topic_id`, BM25 params) |
| POST | `/api/suggest` | Alternative query phrasings |
| POST | `/api/evaluate` | Run evaluation (all_queries / refine / save) |
| GET  | `/api/evaluations?dataset=...` | List saved evaluation runs |
| GET  | `/api/topics?dataset=...` | List topic clusters with top words and sizes |

Full schemas at `http://localhost:8000/docs` (FastAPI auto-docs).

---

## 8. Project structure

```
.
├── api_gateway/            # FastAPI app + static UI mount
│   └── main.py
├── frontend/               # Dark-theme HTML / CSS / JS
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── Services/
│   ├── DataLoaderService/
│   ├── PreProcessingService/
│   ├── RepresentationService/
│   │   ├── tfidf_strategy.py
│   │   ├── bm25_strategy.py
│   │   ├── embedding_strategy.py
│   │   ├── hybrid_serial_strategy.py
│   │   └── hybrid_parallel_strategy.py
│   ├── IndexingService/
│   ├── RetrievalService/
│   ├── SearchService/
│   ├── QueryRefinementService/
│   ├── EvaluationService/
│   ├── EvaluationStoreService/
│   ├── RegistryService/
│   ├── DocumentStoreService/
│   └── TopicService/
├── data_store/             # *.joblib indexes (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 9. Tech stack

Python 3.12 · FastAPI · scikit-learn · rank_bm25 · sentence-transformers · BERTopic
(HDBSCAN + UMAP) · NLTK + pyspellchecker · ir_datasets · PostgreSQL (psycopg3) · vanilla
HTML/CSS/JS with Chart.js · Docker Compose.
