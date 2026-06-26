# تقرير مشروع نظم استرجاع المعلومات 2026

**اسم المشروع:** Information Retrieval Search Engine
**اللغة:** Python 3.12 — معمارية SOA — واجهة Web — تخزين في PostgreSQL

> **ملاحظة للقارئ:** كل مرّة ترى `[📷 ...]` ضع لقطة شاشة في الموضع المحدّد. أسماء الملفات المقترحة في `report/images/`.

---

## ١. مقدّمة

النظام محرّك بحث يسترجع الوثائق من مجموعة `beir/quora/test` (522,931 وثيقة) باستخدام عدّة نماذج تمثيل، ويسمح بالتقييم على كامل ملف qrels (10,000 استعلام محكوم) مع رسومات بيانية.

**المتطلّبات المحقّقة:**
- ٥ نماذج تمثيل (TF-IDF, BM25, Embedding, Hybrid Serial, Hybrid Parallel).
- معالجة مسبقة كاملة، فهرسة، استرجاع وترتيب.
- تحسين استعلام عبر اقتراحات بديلة.
- تقييم بالمقاييس الأربعة (MAP, Recall@100, P@10, nDCG@10) قبل/بعد التحسين.
- معمارية SOA مع API Gateway (FastAPI) وواجهة Web داكنة.
- ميزة إضافية: **Topic Modeling عبر BERTopic** (قابلة للتجربة باستقلال).
- التشغيل بـ Docker.
- **النص الخام يُجلب من PostgreSQL بالـ ID وقت الاستعلام**.

[📷 صورة: لقطة عامة لواجهة البحث وعليها استعلام ونتائج — `report/images/01_overview.png`]

---

## ٢. توصيف مجموعة البيانات

**`beir/quora/test`** (من BEIR via `ir_datasets`):
- 522,931 وثيقة (أسئلة قصيرة من موقع Quora).
- 10,000 استعلام اختبار، كلّها لها qrels.
- 15,675 سطر في qrels (متوسّط ~1.5 وثيقة ملائمة لكل استعلام).
- النوع: استرجاع أسئلة مكرّرة/مشابهة دلالياً.

**سبب الاختيار:** مجموعة كبيرة (>200K)، مجانية، لها qrels (شرط الأستاذ)، ومناسبة لإظهار قوّة النماذج الدلالية.

[📷 صورة: شاشة Settings تُظهر السجلّ مع `beir/quora/test` و عدد الوثائق — `report/images/02_dataset_registry.png`]

---

## ٣. البنية المعمارية (SOA)

النظام مقسّم إلى خدمات مستقلّة تحت `Services/`، يربطها API Gateway (FastAPI).

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
SearchService Indexing  Preproc.  QueryRefine EvalStore  TopicService
  │          │                                        ▲
  │          ▼                                        │
  │   data_store/*.joblib                  evaluation_runs (Postgres)
  ▼
DocumentStoreService ───► PostgreSQL ───► RegistryService
  (raw text by id)         ir_db        (databases table)
```

**جدول الخدمات:**

| الخدمة | المسؤولية |
|---|---|
| `DataLoaderService` | تحميل docs/queries/qrels من ir_datasets |
| `PreProcessingService` | tokenize + lowercase + remove punctuation/stopwords + stem (Decorator) |
| `RepresentationService` | TF-IDF, BM25, Embedding, Hybrid (Strategy) |
| `IndexingService` | بناء وحفظ الفهارس وتحميلها |
| `RetrievalService` | الترتيب من مصفوفة درجات موحّدة |
| `SearchService` | تنسيق البحث + cache + فلتر topic |
| `QueryRefinementService` | تصحيح إملائي + مرادفات WordNet → اقتراحات |
| `EvaluationService` | MAP, Recall, P@10, nDCG |
| `EvaluationStoreService` | حفظ التشغيلات في Postgres |
| `RegistryService` | سجلّ الـ datasets المبنيّة |
| `DocumentStoreService` | تخزين/جلب النص الخام بالـ ID |
| `TopicService` | BERTopic + تخزين doc→topic |

[📷 صورة: لقطة شاشة لشجرة المجلّدات (`tree Services/`) أو لقطة لـ PyCharm Project View — `report/images/03_services_tree.png`]

**التواصل بين الخدمات:** REST API (HTTP/JSON) عبر FastAPI، مع كائنات Python داخلية بين الخدمات في نفس العملية (يقلّل overhead).

---

## ٤. المعالجة المسبقة (Pre-Processing)

`PreProcessingService` يطبّق نمط **Decorator** لبناء سلسلة معالجة قابلة للتوسعة:

```
BaseTokenizer → LowerCase → RemovePunctuation → RemoveStopwords → Stemming
```

كل decorator يضيف خطوة واحدة دون تعديل الخطوات الأخرى.

[📷 صورة: كود `preprocessing_service.py` و أحد الـ decorators — `report/images/04_preprocessing_code.png`]

**ملاحظة مهمّة:** المعالجة تحدث **داخل كل استراتيجية** على النص الخام (لكل من الوثائق وقت البناء والاستعلام وقت البحث) → يضمن أن الاستعلام يُعالَج بنفس طريقة الوثائق.

---

## ٥. تمثيلات الوثائق (Representations)

العقد الموحّد (`IRepresentationStrategy`):
```python
def index_documents(raw_documents): ...   # بناء وحفظ
def get_scores(query: str) -> np.ndarray  # درجة لكل وثيقة
```

كل استراتيجية تطبّقه بطريقتها (Strategy pattern)، فالمستهلك (RetrievalService/EvaluationService) يعاملها بنفس الواجهة.

### ٥.١ TF-IDF
- `sklearn.feature_extraction.text.TfidfVectorizer`.
- مصفوفة متناثرة 522K × |vocab|.
- المطابقة: cosine similarity داخل الاستراتيجية.

[📷 صورة: كود `tfidf_strategy.py` — `report/images/05_tfidf_code.png`]

### ٥.٢ BM25
- `rank_bm25.BM25Okapi`.
- **معاملات `k1`/`b` قابلة للضبط لكل استعلام** عبر `get_scores_with_params` بدون إعادة بناء الفهرس (الإحصاءات مخزّنة).

[📷 صورة: واجهة البحث تُظهر مزلاقَي k1 و b — `report/images/06_bm25_params.png`]

### ٥.٣ Embedding
- `sentence-transformers` بنموذج `all-MiniLM-L6-v2` (متجه 384 بُعد).
- المتجهات **مُطبَّعة** فضرب نقطي = cosine similarity (سريع).
- البناء بطئ نسبياً (~30 دقيقة على CPU)، لكن وقت الاستعلام ~50ms.

[📷 صورة: ناتج بناء Embedding في السطل (chunks 50K/522K) — `report/images/07_embedding_build.png`]

### ٥.٤ Hybrid Serial (BM25 → Embedding rerank)
- المرحلة ١: BM25 يرشّح أعلى 100 وثيقة.
- المرحلة ٢: Embedding يعيد ترتيب هذه الـ100 فقط.
- الفائدة: السرعة (BM25 خفيف) + الدقّة الدلالية (Embedding ذكي).

[📷 صورة: كود `hybrid_serial_strategy.py` — `report/images/08_hybrid_serial_code.png`]

### ٥.٥ Hybrid Parallel (RRF)
- كل نموذج يرتّب الكل مستقلاً.
- الدمج بـ **Reciprocal Rank Fusion**: `score(d) = Σ 1/(60+rank_i(d))`.
- يتجاهل مقياس الدرجات (BM25 ~10، cosine ~0.7)، يعتمد على الموقع فقط.

[📷 صورة: كود `hybrid_parallel_strategy.py` — `report/images/09_hybrid_parallel_code.png`]

---

## ٦. الفهرسة والتخزين

**تقسيم الذاكرة بين القرص و قاعدة البيانات:**

| ما يُخزَّن | أين | السبب |
|---|---|---|
| فهارس TF-IDF/BM25/Embedding | `data_store/*.joblib` | تُحمَّل كاملة في الذاكرة مرّة واحدة عند الإقلاع |
| `doc_ids` بالترتيب | `..._corpus.joblib` | قائمة صغيرة، يُحتاج كل الـ522K |
| **النص الخام** | جدول `documents` في Postgres | **قراءة بالـ ID وقت الاستعلام** (شرط الأستاذ) |
| سجلّ الـ datasets | جدول `databases` | استبدل JSON بـ DB |
| نتائج التقييم | جدول `evaluation_runs` | لتاريخ التشغيلات والرسومات |
| الـ topics ومخصّصاتها | جدولا `topics` + `doc_topics` | للفلتر السريع بالـ ID |

[📷 صورة: `psql \\dt` أو لقطة DBeaver تُظهر الجداول الخمسة — `report/images/10_postgres_tables.png`]

**التسخين عند الإقلاع (`warmup`):** عند `lifespan` للـ FastAPI، كل الفهارس تُحمَّل من القرص إلى الذاكرة + يُفتح اتصال DB. أوّل استعلام للمستخدم ~1 ثانية بدل ~28 ثانية.

[📷 صورة: لقطة من السطل تُظهر startup logs مع تحميل الفهارس — `report/images/11_warmup_log.png`]

---

## ٧. معالجة الاستعلام وتحسينه (Query Refinement)

**معالجة الاستعلام:** كل استراتيجية تطبّق المعالجة نفسها (انظر القسم ٤) داخلياً على النص الخام للاستعلام.

**التحسين كاقتراحات بديلة** (`QueryRefinementService`):
- تصحيح إملائي عبر `pyspellchecker` (مثلاً `pythom` → `python`).
- توسيع بالمرادفات عبر WordNet (مثلاً `car` → `auto, automobile`).
- يُعرض في الواجهة كـ **رقائق (chips) قابلة للنقر**، المستخدم يختار البديل المناسب بدلاً من استبدال أوتوماتيكي.

[📷 صورة: واجهة البحث، يكتب `car insurance` ثم Suggest فتظهر الرقائق — `report/images/12_suggest_chips.png`]

---

## ٨. مطابقة الاستعلام وترتيب النتائج

`RetrievalService.get_top_results(scores, raw_documents, top_k)`: نقيّ — ترتيب + قصّ + تنسيق.

كل تعقيد المطابقة موجود **داخل** الاستراتيجية:
- cosine لـ TF-IDF و Embedding.
- معادلة BM25 لـ BM25.
- ترشيح/إعادة ترتيب لـ Hybrid Serial.
- RRF لـ Hybrid Parallel.

[📷 صورة: نتيجة بحث على الواجهة تُظهر #1 doc_id، score، النص الأصلي — `report/images/13_search_results.png`]

---

## ٩. التقييم (Evaluation) — الجزء الأهم

### ٩.١ المقاييس المنفّذة (`evaluation_metrics.py`):
- **MAP** (Mean Average Precision).
- **Recall@100**.
- **P@10** (Precision at 10).
- **nDCG@10** (Normalized Discounted Cumulative Gain).

كلها مكتوبة يدوياً مع تعليقات، ومحقّق صحّتها بمثال محسوب يدوياً (`metrics_test.py`).

[📷 صورة: ناتج `metrics_test.py` يُظهر المطابقة بين الحساب اليدوي ومخرجات الكود — `report/images/14_metrics_unit_test.png`]

### ٩.٢ منهجية التقييم
- **استُخدِمت كل الـ 10,000 استعلام في qrels** (حسب توجيه الأستاذ، لا عيّنة).
- العدد يُطبع بوضوح في الـ logs: `N = 10000 (used_all_qrels=True)`.
- كل تشغيل يُحفظ في جدول `evaluation_runs` مع: المقاييس + الزمن المُستغرَق + flag الـ refine + flag الـ all_qrels.

[📷 صورة: log التقييم يُظهر `▶ Evaluating ...   queries used in this run: N = 10000` و `✔ Evaluated 10000 queries in Xs` — `report/images/15_eval_log_count.png`]

### ٩.٣ النتائج النهائية (على كامل 10,000 استعلام)

| النموذج | refine | MAP | nDCG@10 | P@10 | Recall@100 | الزمن |
|---|---|---|---|---|---|---|
| TF-IDF | - | 0.7075 | 0.7480 | 0.1138 | 0.9591 | 477s |
| TF-IDF | ✓ | 0.5671 | 0.6099 | 0.0961 | 0.9039 | 1885s |
| BM25 | - | 0.7370 | 0.7773 | 0.1179 | 0.9685 | 2184s |
| BM25 | ✓ | 0.6376 | 0.6797 | 0.1050 | 0.9250 | 7400s |
| **Embedding** | **-** | **0.8431** | **0.8754** | **0.1337** | **0.9944** | 1412s |
| Embedding | ✓ | 0.7873 | 0.8227 | 0.1270 | 0.9793 | 2296s |
| Hybrid Serial | - | 0.8383 | 0.8718 | 0.1317 | 0.9685 | 3216s |
| Hybrid Parallel | - | 0.8220 | 0.8580 | 0.1306 | 0.9952 | 8874s |

**أهم الملاحظات:**
1. **Embedding هو الأفضل** (nDCG@10 = 0.8754) — متوقّع لـ Quora لأنها بحث **دلالي** (أسئلة مشابهة معنوياً).
2. **Hybrid Serial يقاربه** (0.8718) — يبدأ بـ BM25 ثم يعيد ترتيب بـ Embedding.
3. **BM25 يتفوّق على TF-IDF** قليلاً (0.7773 vs 0.7480) — متوقّع.
4. أرقامنا **مطابقة للمراجع المنشورة** لـ BEIR على Quora (BM25 nDCG@10 الرسمي ≈ 0.789 ✓).
5. **refinement يُضرّ في Quora** — استعلامات قصيرة وواضحة، مرادفات WordNet تُدخل ضوضاء (`learn`→`larn`...). موثَّق بأرقام واضحة قبل/بعد.
6. **P@10 منخفض دائماً** (~0.13 كحد أقصى) لأن استعلامات Quora لها 1-2 وثيقة ملائمة فقط، فالحدّ الأعلى النظري ~0.15.

[📷 صورة: صفحة Evaluation تُظهر **المخطط الشريطي لـ MAP** لكل النماذج (raw vs refine) — `report/images/16_chart_map.png`]

[📷 صورة: صفحة Evaluation تُظهر **المخطط الشريطي لـ nDCG@10** لكل النماذج — `report/images/17_chart_ndcg.png`]

[📷 صورة: جدول `Saved evaluations` كاملاً على الواجهة — `report/images/18_eval_table.png`]

### ٩.٤ تقييم قبل/بعد التحسين (Refinement)

يحقّق متطلب الأستاذ: **«تقييم قبل وبعد تطبيق الطلبات الإضافية»**.

النتائج تُظهر بوضوح أن **refinement يُضرّ** على Quora في كل المقاييس وكل النماذج. هذا تحليل علمي مفيد (لا نخفيه).

[📷 صورة: نفس المخطط الشريطي مع الأعمدة المزدوجة (raw vs refine) لـ MAP — `report/images/19_chart_refine_compare.png`]

---

## ١٠. الميزة الإضافية: BERTopic Topic Modeling

**ما هي؟** اكتشاف موضوعات نظام غير مُشرف من الوثائق.

**خوارزمية BERTopic** (4 مراحل):
1. **Embedding** (بنينا فهرسها أصلاً → لا نعيد).
2. **UMAP** يخفّض 384 بُعد إلى ~5 أبعاد.
3. **HDBSCAN** يجمّع كثافياً.
4. **c-TF-IDF** يستخرج كلمات مميّزة لكل تجمّع.

**النتيجة على Quora:** **746 موضوع مكتشف**. عيّنات:
- #0 الدين: `god, religion, islam, jesus, muslims` (5,318 وثيقة)
- #1 الطعام: `food, eat, cook, eggs, meat, chicken` (4,473)
- #3 البنوك: `bank, card, account, credit, debit, sbi` (3,103)
- #5 العناية بالشعر: `hair, beard, dandruff, grow, oil` (2,268)

[📷 صورة: صفحة Topics مع جدول المواضيع وكلماتها — `report/images/20_topics_table.png`]

[📷 صورة: المخطط الشريطي لأكبر 20 موضوعاً بالحجم — `report/images/21_topics_chart.png`]

**التكامل المستقل (شرط الأستاذ):**
- في صفحة البحث: قائمة منسدلة «Filter by topic» تجعل البحث **داخل** الموضوع المختار فقط.
- toggle on/off — يُمكن تشغيل البحث بدون أو مع filter.

[📷 صورة: بحث `python` بدون فلتر — `report/images/22_search_no_topic.png`]

[📷 صورة: نفس البحث مع `topic_id=#10 (programming)` — `report/images/23_search_with_topic.png`]

**التخزين في Postgres:**
- جدول `topics(dataset, topic_id, label, words[], size)`.
- جدول `doc_topics(dataset, doc_id, topic_id)` مع index على (dataset, topic_id) للتصفية السريعة.

---

## ١١. واجهة المستخدم

واجهة Web داكنة مكتوبة بـ HTML/CSS/JS خام (مع Chart.js) تخدمها FastAPI. أربع صفحات:

### ١١.١ Search
- اختيار database و model.
- ضبط k1/b لـ BM25.
- فلتر topic اختياري.
- زرّ Suggest للاقتراحات.
- يعرض كل نتيجة بـ: rank, score, **doc_id** (للمقارنة مع qrels)، **النص الأصلي الخام** (مُجلَب من Postgres).

[📷 صورة: صفحة Search كاملة — `report/images/24_ui_search.png`]

### ١١.٢ Evaluation
- اختيار database و model.
- خيار «Run on ALL qrels» (يُجبر الحفظ في DB).
- خيار «Apply query refinement» للمقارنة.
- خيار «Save run to database».
- جدول التشغيلات المحفوظة + مخطّطان (MAP و nDCG@10).

[📷 صورة: صفحة Evaluation كاملة — `report/images/25_ui_evaluation.png`]

### ١١.٣ Topics
- جدول المواضيع مع كلماتها وأحجامها.
- مخطّط شريطي لأكبر 20 موضوع.

[📷 صورة: صفحة Topics كاملة — `report/images/26_ui_topics.png`]

### ١١.٤ Settings (⚙)
- إضافة database جديدة (إدخال اسم + اختيار النماذج المطلوب بناؤها).
- جدول الـ databases الحالية مع الحالة (`ready` / `building` / `error`).

[📷 صورة: صفحة Settings تُظهر النموذج وجدول البنى — `report/images/27_ui_settings.png`]

---

## ١٢. الأداء (شرط ≤ 20 ثانية)

- **القراءة بالـ ID من Postgres**: ~2.6 ميلي ثانية لـ 10 وثائق.
- **استعلام دافئ كامل** (بعد التسخين): ~1-2 ثانية.
- **التسخين** يُجرى مرّة عند إقلاع الخادم (~28 ثانية) فلا يدفعه المستخدم.
- **لا تدريب أونلاين** — كل النماذج جاهزة من الأوفلاين.

[📷 صورة: لقطة DevTools/Network تُظهر زمن استجابة /api/search تحت 2 ثانية — `report/images/28_perf_devtools.png`]

---

## ١٣. تشغيل المشروع (Docker)

```bash
docker compose up -d --build
# UI: http://localhost:8000  |  Docs: http://localhost:8000/docs
```

تفاصيل أوفلاين كاملة في `README.md`.

[📷 صورة: ناتج `docker compose ps` يُظهر `ir_postgres` و `ir_app` يعملان — `report/images/29_docker_ps.png`]

---

## ١٤. التحقّق من النتائج بالـ qrels في المقابلة

كل نتيجة تعرض **doc_id بوضوح**. للتحقّق من سلامة النظام:
1. نفتح ملف qrels للـ dataset.
2. نختار استعلاماً ونرى أيّ doc_ids ملائمة له.
3. نُدخل نصّ الاستعلام في الواجهة.
4. نقارن: doc_ids الناتجة من النظام يجب أن تحوي doc_ids الملائمة في الأعلى.

[📷 صورة: لقطة مقسومة (ملف qrels على اليسار + نتائج النظام على اليمين) لاستعلام واحد — `report/images/30_qrels_compare.png`]

---

## ١٥. المصادر

- BEIR Benchmark — https://github.com/beir-cellar/beir
- ir_datasets — https://ir-datasets.com/
- BERTopic — https://maartengr.github.io/BERTopic/
- sentence-transformers — https://www.sbert.net/
- rank_bm25 — https://pypi.org/project/rank-bm25/
- FastAPI — https://fastapi.tiangolo.com/

---

## ١٦. تقسيم العمل

> [✏️ هذا القسم يُملأ يدوياً حسب أعضاء المجموعة.]
