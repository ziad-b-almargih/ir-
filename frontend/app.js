const API = "/api";

// ----------------------------- helpers -----------------------------
async function apiGet(path) {
    const res = await fetch(API + path);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    return res.json();
}

async function apiPost(path, body) {
    const res = await fetch(API + path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    return data;
}

function fillSelect(select, items) {
    select.innerHTML = "";
    items.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        select.appendChild(opt);
    });
}

// ----------------------------- navigation -----------------------------
document.querySelectorAll(".navbtn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".navbtn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("view-" + btn.dataset.view).classList.add("active");
        if (btn.dataset.view === "settings") loadDatabases();
        if (btn.dataset.view === "topics") loadTopicsView();
    });
});

// ----------------------------- datasets / models -----------------------------
async function loadDatasets() {
    const {datasets} = await apiGet("/datasets");
    fillSelect(document.getElementById("searchDataset"), datasets);
    fillSelect(document.getElementById("evalDataset"), datasets);
    fillSelect(document.getElementById("topicsDataset"), datasets);
    if (datasets.length) {
        await loadModels(datasets[0], "searchModel");
        await loadModels(datasets[0], "evalModel");
        await loadTopicsForFilter(datasets[0]);
        toggleBm25();
        loadQrels();
    }
}

let _allTopics = [];   // cached full topic list so the keyword filter can rebuild quickly

function renderTopicOptions(topics) {
    const select = document.getElementById("topicFilter");
    const previous = select.value;
    select.innerHTML = '<option value="">All topics</option>';
    topics.forEach(t => {
        const opt = document.createElement("option");
        opt.value = t.topic_id;
        const words = (t.words || []).slice(0, 4).join(", ");
        opt.textContent = `#${t.topic_id} · ${words || "outliers"} (${t.size})`;
        select.appendChild(opt);
    });
    // Keep the user's previous selection if it still matches a visible topic.
    if ([...select.options].some(o => o.value === previous)) select.value = previous;
}

async function loadTopicsForFilter(dataset) {
    // Populate the topic filter dropdown for the Search view; quietly skip if topics aren't built.
    _allTopics = [];
    renderTopicOptions(_allTopics);
    document.getElementById("topicSearch").value = "";
    if (!dataset) return;
    try {
        const {topics} = await apiGet(`/topics?dataset=${encodeURIComponent(dataset)}`);
        _allTopics = topics;
        renderTopicOptions(_allTopics);
    } catch (err) { /* topics not built yet — fine, filter just stays empty */ }
}

document.getElementById("topicSearch").addEventListener("input", e => {
    const needle = e.target.value.trim().toLowerCase();
    if (!needle) { renderTopicOptions(_allTopics); return; }
    // Match on any of the topic's top words OR its id.
    const filtered = _allTopics.filter(t =>
        String(t.topic_id) === needle ||
        (t.words || []).some(w => w.toLowerCase().includes(needle)) ||
        (t.label || "").toLowerCase().includes(needle));
    renderTopicOptions(filtered);
});

async function loadModels(dataset, targetId) {
    if (!dataset) return;
    const {models} = await apiGet(`/datasets/${dataset}/models`);
    fillSelect(document.getElementById(targetId), models);
}

function toggleBm25() {
    const isBm25 = document.getElementById("searchModel").value === "bm25";
    document.getElementById("bm25Params").classList.toggle("show", isBm25);
}

document.getElementById("searchDataset").addEventListener("change", async e => {
    await loadModels(e.target.value, "searchModel");
    await loadTopicsForFilter(e.target.value);
    toggleBm25();
    qrelsOffset = 0;
    qrelsQuery = "";
    document.getElementById("qrelFilter").value = "";
    loadQrels();
});
document.getElementById("searchModel").addEventListener("change", toggleBm25);
document.getElementById("evalDataset").addEventListener("change", e =>
    loadModels(e.target.value, "evalModel"));

// ----------------------------- search -----------------------------
async function doSearch() {
    const note = document.getElementById("searchNote");
    const resultsDiv = document.getElementById("results");
    const query = document.getElementById("query").value.trim();
    if (!query) { note.className = "note err"; note.textContent = "Please enter a query."; return; }

    const model = document.getElementById("searchModel").value;
    const payload = {
        dataset: document.getElementById("searchDataset").value,
        model: model,
        query: query,
        top_k: parseInt(document.getElementById("topK").value, 10),
        refine: document.getElementById("searchRefine").checked,
        use_history: document.getElementById("searchUseHistory").checked,
    };
    if (model === "bm25") {
        payload.k1 = parseFloat(document.getElementById("k1").value);
        payload.b = parseFloat(document.getElementById("b").value);
    }
    const topicVal = document.getElementById("topicFilter").value;
    if (topicVal !== "") payload.topic_id = parseInt(topicVal, 10);

    note.className = "note"; note.textContent = "Searching...";
    resultsDiv.innerHTML = "";
    try {
        // حساب الوقت قبل الاستدعاء
        const startTime = performance.now();

        const data = await apiPost("/search", payload);

        // حساب الوقت بعد الاستدعاء
        const timeTaken = ((performance.now() - startTime) / 1000).toFixed(2); // التحويل لثواني

        note.className = "note ok";
        let extra = "";
        if (data.history_expansion && data.history_expansion.length) {
            extra = ` — history added: [${data.history_expansion.join(", ")}]`;
        } else if (payload.use_history) {
            extra = " — history: no similar past queries found";
        } else if (data.query_used && data.query_used !== query) {
            extra = ` — query used: "${data.query_used}"`;
        }
        note.textContent = `${data.count} results (in ${timeTaken} seconds)${extra}`;

        document.getElementById("searchMetrics").style.display = "none";
        renderResults(data.results, resultsDiv);
    } catch (err) {
        note.className = "note err"; note.textContent = "Search failed: " + err.message;
    }
}

// Long text is collapsed by default with a "Show more" toggle (user-requested behaviour).
const COLLAPSE_LIMIT = 220;
function renderResults(results, container) {
    container.innerHTML = "";
    if (!results.length) {
        container.innerHTML = '<span class="muted">No matching documents found.</span>';
        return;
    }
    results.forEach(r => {
        const div = document.createElement("div");
        div.className = "result";

        // Optional relevance pill when results come from /api/test-query.
        let pill = "";
        if (r.relevant === true) pill = `<span class="pill rel">relevant (grade ${r.grade ?? 1})</span>`;
        else if (r.relevant === false) pill = `<span class="pill miss">not in qrels</span>`;

        div.innerHTML = `<div class="meta">
                <span class="rank">#${r.rank}</span> ·
                score <span class="score">${r.score}</span> ·
                doc_id <code>${r.doc_id}</code>
                ${pill}
            </div>
            <div class="body collapsed"></div>
            <span class="toggle" style="display:none">▾ Show more</span>`;
        const body = div.querySelector(".body");
        const toggle = div.querySelector(".toggle");
        body.textContent = r.text || "";
        if ((r.text || "").length > COLLAPSE_LIMIT) {
            toggle.style.display = "inline-block";
            toggle.addEventListener("click", () => {
                const isCollapsed = body.classList.toggle("collapsed");
                toggle.textContent = isCollapsed ? "▾ Show more" : "▴ Show less";
            });
        } else {
            body.classList.remove("collapsed");   // short text: no need to clip
        }
        container.appendChild(div);
    });
}

function renderSearchMetrics(metrics) {
    // Per-query metrics card shown above the results when /api/test-query is used.
    const container = document.getElementById("searchMetrics");
    container.innerHTML = "";
    container.style.display = "grid";
    const order = ["AP", "nDCG@10", "P@10", "Recall@100", "relevant_in_qrels"];
    order.forEach(key => {
        if (metrics[key] === undefined || metrics[key] === null) return;
        const div = document.createElement("div");
        div.className = "metric";
        const label = key === "relevant_in_qrels" ? "Relevant in qrels" : key;
        div.innerHTML = `<div class="k">${label}</div><div class="v">${metrics[key]}</div>`;
        container.appendChild(div);
    });
}

async function doTestQuery(queryId) {
    // Submits one of the dataset's judged queries and shows per-query metrics + results.
    const note = document.getElementById("searchNote");
    const resultsDiv = document.getElementById("results");
    const model = document.getElementById("searchModel").value;
    const payload = {
        dataset: document.getElementById("searchDataset").value,
        model,
        query_id: String(queryId),
        top_k: parseInt(document.getElementById("topK").value, 10),
    };
    if (model === "bm25") {
        payload.k1 = parseFloat(document.getElementById("k1").value);
        payload.b = parseFloat(document.getElementById("b").value);
    }
    const topicVal = document.getElementById("topicFilter").value;
    if (topicVal !== "") payload.topic_id = parseInt(topicVal, 10);

    note.className = "note"; note.textContent = `Testing qrel #${queryId}...`;
    resultsDiv.innerHTML = "";
    document.getElementById("searchMetrics").style.display = "none";

    try {
        const data = await apiPost("/test-query", payload);
        document.getElementById("query").value = data.query_text || "";
        renderSearchMetrics(data.metrics || {});
        renderResults(data.results || [], resultsDiv);
        note.className = "note ok";
        note.textContent = `Tested qrel #${data.query_id} · ${data.metrics.relevant_in_qrels} relevant doc(s) in qrels.`;
        window.scrollTo({top: 0, behavior: "smooth"});
    } catch (err) {
        note.className = "note err"; note.textContent = "Test failed: " + err.message;
    }
}

// Show alternative query phrasings as clickable chips; clicking one runs the search.
async function showSuggestions() {
    const chips = document.getElementById("suggestions");
    const query = document.getElementById("query").value.trim();
    chips.innerHTML = "";
    if (!query) return;
    try {
        const data = await apiPost("/suggest", {query});
        if (!data.suggestions.length) {
            chips.innerHTML = '<span class="muted">No suggestions.</span>';
            return;
        }
        data.suggestions.forEach(s => {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.innerHTML = '<span class="tag">try</span>';
            chip.appendChild(document.createTextNode(s));
            chip.addEventListener("click", () => {
                document.getElementById("query").value = s;
                chips.innerHTML = "";
                doSearch();
            });
            chips.appendChild(chip);
        });
    } catch (err) {
        chips.innerHTML = `<span class="muted">Suggest failed: ${err.message}</span>`;
    }
}

document.getElementById("searchBtn").addEventListener("click", doSearch);
document.getElementById("suggestBtn").addEventListener("click", showSuggestions);
document.getElementById("query").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

// ----------------------------- evaluation -----------------------------
// "All qrels" forces save=true (per professor's rule: only full runs are authoritative).
const allQueriesEl = document.getElementById("allQueries");
const evalSaveEl = document.getElementById("evalSave");
const numQueriesEl = document.getElementById("numQueries");
allQueriesEl.addEventListener("change", () => {
    if (allQueriesEl.checked) {
        evalSaveEl.checked = true; evalSaveEl.disabled = true;
        numQueriesEl.disabled = true;
    } else {
        evalSaveEl.disabled = false;
        numQueriesEl.disabled = false;
    }
});

function renderMetricCards(data) {
    const metricsDiv = document.getElementById("metrics");
    metricsDiv.innerHTML = "";
    const order = ["MAP", "Recall@100", "P@10", "nDCG@10", "elapsed_seconds"];
    order.forEach(key => {
        if (!(key in data) || data[key] === null || data[key] === undefined) return;
        const div = document.createElement("div");
        div.className = "metric";
        const label = key === "elapsed_seconds" ? "Time (s)" : key;
        div.innerHTML = `<div class="k">${label}</div><div class="v">${data[key]}</div>`;
        metricsDiv.appendChild(div);
    });
}

document.getElementById("evalBtn").addEventListener("click", async () => {
    const note = document.getElementById("evalNote");
    const payload = {
        dataset: document.getElementById("evalDataset").value,
        model: document.getElementById("evalModel").value,
        num_queries: parseInt(numQueriesEl.value, 10),
        all_queries: allQueriesEl.checked,
        refine: document.getElementById("evalRefine").checked,
        save: evalSaveEl.checked,
    };

    note.className = "note";
    note.textContent = payload.all_queries
        ? "Started in the background (full qrels can take hours). Refresh saved runs to see progress."
        : "Evaluating...";
    document.getElementById("metrics").innerHTML = "";

    try {
        const data = await apiPost("/evaluate", payload);
        if (payload.all_queries) {
            note.className = "note ok";
            note.textContent = `Queued (run #${data.run_id}). ${data.message}`;
        } else {
            note.className = "note ok";
            note.textContent = `Evaluated ${data.queries_evaluated} queries in ${data.elapsed_seconds}s.`;
            renderMetricCards(data);
        }
        loadEvaluations();
    } catch (err) {
        note.className = "note err"; note.textContent = "Evaluation failed: " + err.message;
    }
});

document.getElementById("refreshEvalBtn").addEventListener("click", loadEvaluations);
document.getElementById("evalDataset").addEventListener("change", loadEvaluations);

async function loadEvaluations() {
    const dataset = document.getElementById("evalDataset").value;
    if (!dataset) return;
    try {
        const {evaluations} = await apiGet(`/evaluations?dataset=${encodeURIComponent(dataset)}`);
        const rows = document.getElementById("evalRows");
        rows.innerHTML = "";
        evaluations.forEach(e => {
            const tr = document.createElement("tr");
            const cell = v => v === null || v === undefined ? "-" : v;
            tr.innerHTML = `<td>${e.id}</td><td>${e.model}</td>
                <td>${e.refine ? "✓" : ""}</td><td>${e.used_all_qrels ? "✓" : ""}</td>
                <td>${e.queries_evaluated}</td>
                <td>${cell(e.MAP)}</td><td>${cell(e["nDCG@10"])}</td>
                <td>${cell(e["P@10"])}</td><td>${cell(e["Recall@100"])}</td>
                <td>${cell(e.elapsed_seconds)}</td>
                <td><span class="badge ${e.status}">${e.status}</span></td>
                <td class="when">${(e.created_at || "").replace("T"," ")}</td>`;
            rows.appendChild(tr);
        });
        renderCharts(evaluations);
    } catch (err) { /* dataset list may not be ready yet */ }
}

// Bar charts: latest READY run per (model, refine) combo, for MAP, nDCG@10, P@10, Recall@100.
let chartMap = null, chartNdcg = null, chartP10 = null, chartRecall = null;
function renderCharts(evaluations) {
    const latest = {};   // key = `${model}|${refine}` -> evaluation
    evaluations.filter(e => e.status === "ready").forEach(e => {
        const key = `${e.model}|${e.refine ? "refine" : "raw"}`;
        if (!(key in latest)) latest[key] = e;   // list is desc by id, so first is latest
    });
    const entries = Object.values(latest);
    const models = [...new Set(entries.map(e => e.model))].sort();
    const groupFor = refine => models.map(m => {
        const hit = entries.find(e => e.model === m && (e.refine === refine));
        return hit ? hit : null;
    });
    const datasetsFor = key => [
        {label: "raw",    data: groupFor(false).map(e => e ? e[key] : null), backgroundColor: "#73839c"},
        {label: "refine", data: groupFor(true).map(e  => e ? e[key] : null), backgroundColor: "#f5a623"},
    ];
    const baseOpts = {
        responsive: true,
        plugins: {legend: {labels: {color: "#cdd8e8"}}},
        scales: {
            x: {ticks: {color: "#cdd8e8"}, grid: {color: "#26324a"}},
            y: {ticks: {color: "#cdd8e8"}, grid: {color: "#26324a"}, beginAtZero: true},
        },
    };
    const titled = title => ({...baseOpts, plugins: {...baseOpts.plugins,
        title: {display: true, text: title, color: "#cdd8e8"}}});
    if (chartMap) chartMap.destroy();
    if (chartNdcg) chartNdcg.destroy();
    if (chartP10) chartP10.destroy();
    if (chartRecall) chartRecall.destroy();
    chartMap = new Chart(document.getElementById("chartMap"),
        {type: "bar", data: {labels: models, datasets: datasetsFor("MAP")},      options: titled("MAP")});
    chartNdcg = new Chart(document.getElementById("chartNdcg"),
        {type: "bar", data: {labels: models, datasets: datasetsFor("nDCG@10")},  options: titled("nDCG@10")});
    chartP10 = new Chart(document.getElementById("chartP10"),
        {type: "bar", data: {labels: models, datasets: datasetsFor("P@10")},     options: titled("P@10")});
    chartRecall = new Chart(document.getElementById("chartRecall"),
        {type: "bar", data: {labels: models, datasets: datasetsFor("Recall@100")}, options: titled("Recall@100")});
}

// ----------------------------- settings: add / list databases -----------------------------
document.getElementById("buildBtn").addEventListener("click", async () => {
    const note = document.getElementById("buildNote");
    const dataset = document.getElementById("newDataset").value.trim();
    const models = [...document.querySelectorAll(".modelChk:checked")].map(c => c.value);
    if (!dataset) { note.className = "note err"; note.textContent = "Enter a dataset name."; return; }
    if (!models.length) { note.className = "note err"; note.textContent = "Pick at least one model."; return; }

    note.className = "note"; note.textContent = "Build started in background...";
    try {
        const startTime = performance.now();

        await apiPost("/databases", {dataset, models});

        const timeTaken = ((performance.now() - startTime) / 1000).toFixed(2);

        note.className = "note ok";
        // إضافة الوقت المستغرق لبدء العملية
        note.textContent = `Started building '${dataset}' in ${timeTaken}s. Refresh to see status.`;

        loadDatabases();
    } catch (err) {
        note.className = "note err"; note.textContent = "Build failed: " + err.message;
    }
});
document.getElementById("refreshBtn").addEventListener("click", loadDatabases);

async function loadDatabases() {
    const rows = document.getElementById("dbRows");
    try {
        const {databases} = await apiGet("/databases");
        rows.innerHTML = "";
        databases.forEach(db => {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${db.dataset}</td><td>${(db.models || []).join(", ")}</td>
                <td>${db.doc_count || 0}</td>
                <td><span class="badge ${db.status}">${db.status}</span></td>
                <td>${(db.updated_at || "").replace("T", " ")}</td>`;
            rows.appendChild(tr);
        });
        // Refresh the search/eval dataset lists too (a build may have finished).
        loadDatasets();
    } catch (err) {
        rows.innerHTML = `<tr><td colspan="5" class="muted">${err.message}</td></tr>`;
    }
}

// ----------------------------- qrels sidebar -----------------------------
const QRELS_PAGE = 25;
let qrelsOffset = 0;
let qrelsQuery = "";

async function loadQrels() {
    const dataset = document.getElementById("searchDataset").value;
    const listEl = document.getElementById("qrelsList");
    if (!dataset) {
        listEl.innerHTML = '<span class="muted">Pick a database first.</span>';
        document.getElementById("qrelsCount").textContent = "—";
        return;
    }
    listEl.innerHTML = '<span class="muted">Loading...</span>';
    try {
        const params = new URLSearchParams({
            dataset, q: qrelsQuery, limit: QRELS_PAGE, offset: qrelsOffset});
        const data = await apiGet(`/qrels?${params.toString()}`);
        document.getElementById("qrelsCount").textContent =
            `${data.queries.length ? data.offset + 1 : 0}–${data.offset + data.queries.length} of ${data.total}`;
        listEl.innerHTML = "";
        if (!data.queries.length) {
            listEl.innerHTML = '<span class="muted">No matching queries.</span>';
            return;
        }
        data.queries.forEach(q => listEl.appendChild(renderQrel(q, dataset)));
    } catch (err) {
        listEl.innerHTML = `<span class="muted">Failed: ${err.message}</span>`;
    }
}

function renderQrel(q, dataset) {
    const wrapper = document.createElement("div");
    wrapper.className = "qrel";
    wrapper.innerHTML = `
        <div class="qrel-head">
            <span class="qid">#${q.query_id}</span>
            <span class="qtext"></span>
            <span class="arrow">▶</span>
        </div>
        <div class="qrel-body">
            <div class="qrel-docs muted">Loading relevant documents...</div>
            <div class="qrel-actions">
                <button class="btn btn-primary qrel-test" style="padding:.35rem .8rem">Test</button>
                <button class="btn qrel-fill" style="padding:.35rem .8rem">Fill only</button>
            </div>
        </div>`;
    wrapper.querySelector(".qtext").textContent = q.text;

    wrapper.querySelector(".qrel-head").addEventListener("click", async () => {
        wrapper.classList.toggle("open");
        if (wrapper.classList.contains("open") && !wrapper.dataset.loaded) {
            await fetchRelevantDocs(wrapper, dataset, q);
            wrapper.dataset.loaded = "1";
        }
    });
    wrapper.querySelector(".qrel-test").addEventListener("click", e => {
        e.stopPropagation();
        doTestQuery(q.query_id);
    });
    wrapper.querySelector(".qrel-fill").addEventListener("click", e => {
        e.stopPropagation();
        document.getElementById("query").value = q.text;
        document.getElementById("query").focus();
    });
    return wrapper;
}

async function fetchRelevantDocs(wrapper, dataset, q) {
    const container = wrapper.querySelector(".qrel-docs");
    container.innerHTML = "";
    const ids = q.doc_ids.slice(0, 30);   // cap to keep the panel responsive
    if (!ids.length) {
        container.innerHTML = '<span class="muted">No relevant documents listed.</span>';
        return;
    }
    try {
        const params = new URLSearchParams({dataset, ids: ids.join(",")});
        const data = await apiGet(`/documents?${params.toString()}`);
        container.innerHTML = "";
        data.documents.forEach(doc => {
            const grade = q.relevance ? q.relevance[doc.doc_id] : "";
            const card = document.createElement("div");
            card.className = "qrel-doc";
            card.innerHTML = `<span class="did">${doc.doc_id}</span>` +
                (grade !== "" && grade !== undefined ? `<span class="grade">rel=${grade}</span>` : "") +
                `<span class="dtext"></span>`;
            card.querySelector(".dtext").textContent = " " + doc.text;
            container.appendChild(card);
        });
        if (q.doc_ids.length > ids.length) {
            const more = document.createElement("span");
            more.className = "muted";
            more.style.fontSize = ".75rem";
            more.textContent = `(+${q.doc_ids.length - ids.length} more relevant docs)`;
            container.appendChild(more);
        }
    } catch (err) {
        container.innerHTML = `<span class="muted">Failed: ${err.message}</span>`;
    }
}

document.getElementById("qrelFilter").addEventListener("input", e => {
    qrelsQuery = e.target.value.trim();
    qrelsOffset = 0;
    clearTimeout(window._qrelsTimer);
    window._qrelsTimer = setTimeout(loadQrels, 250);   // debounce
});
document.getElementById("qrelsPrev").addEventListener("click", () => {
    qrelsOffset = Math.max(0, qrelsOffset - QRELS_PAGE);
    loadQrels();
});
document.getElementById("qrelsNext").addEventListener("click", () => {
    qrelsOffset += QRELS_PAGE;
    loadQrels();
});

// ----------------------------- topics view -----------------------------

document.getElementById("buildTopicBtn").addEventListener("click", async () => {
    const dataset = document.getElementById("topicsDataset").value;
    if (!dataset) {
        alert("Please select a dataset first.");
        return;
    }

    const btn = document.getElementById("buildTopicBtn");
    const originalText = btn.textContent;

    // Update button UI to show it's working
    btn.textContent = "Building... (This may take a while)";
    btn.disabled = true;

    try {
        const startTime = performance.now();

        // Call the API
        await apiPost("/build-topic", { dataset: dataset });

        const timeTaken = ((performance.now() - startTime) / 1000).toFixed(2);

        // Let the user know it finished
        alert(`Topics built successfully for '${dataset}' in ${timeTaken} seconds!`);

        // Refresh the table and chart to show the new topics
        await loadTopicsView();
    } catch (err) {
        alert("Failed to build topics: " + err.message);
    } finally {
        // Reset button UI
        btn.textContent = originalText;
        btn.disabled = false;
    }
});

let chartTopicSizes = null;

async function loadTopicsView() {
    const dataset = document.getElementById("topicsDataset").value;
    if (!dataset) return;
    const rows = document.getElementById("topicRows");
    rows.innerHTML = '<tr><td colspan="4" class="muted">Loading...</td></tr>';
    try {
        const {topics} = await apiGet(`/topics?dataset=${encodeURIComponent(dataset)}`);
        rows.innerHTML = "";
        if (!topics.length) {
            rows.innerHTML = '<tr><td colspan="4" class="muted">No topics built for this dataset.</td></tr>';
            if (chartTopicSizes) { chartTopicSizes.destroy(); chartTopicSizes = null; }
            return;
        }
        topics.forEach(t => {
            const tr = document.createElement("tr");
            const wordsText = (t.words || []).slice(0, 10).join(", ");
            tr.innerHTML = `<td>${t.topic_id}</td><td>${t.label || ""}</td>
                <td>${t.size}</td><td>${wordsText || "<em class='muted'>outliers</em>"}</td>`;
            rows.appendChild(tr);
        });

        // Bar chart: top 20 topics by size, skipping the outlier bucket (-1) for clarity.
        const ranked = topics.filter(t => t.topic_id !== -1).slice(0, 20);
        const labels = ranked.map(t => {
            const w = (t.words || []).slice(0, 2).join(",");
            return `#${t.topic_id} ${w}`;
        });
        const data = ranked.map(t => t.size);
        if (chartTopicSizes) chartTopicSizes.destroy();
        chartTopicSizes = new Chart(document.getElementById("chartTopicSizes"), {
            type: "bar",
            data: {labels, datasets: [{label: "Documents per topic", data,
                                        backgroundColor: "#f5a623"}]},
            options: {
                responsive: true,
                plugins: {legend: {labels: {color: "#cdd8e8"}},
                          title: {display: true, text: "Top topics by size", color: "#cdd8e8"}},
                scales: {
                    x: {ticks: {color: "#cdd8e8", maxRotation: 60, minRotation: 30},
                        grid: {color: "#26324a"}},
                    y: {ticks: {color: "#cdd8e8"}, grid: {color: "#26324a"}, beginAtZero: true},
                },
            },
        });
    } catch (err) {
        rows.innerHTML = `<tr><td colspan="4" class="muted">${err.message}</td></tr>`;
    }
}

document.getElementById("refreshTopicsBtn").addEventListener("click", loadTopicsView);
document.getElementById("topicsDataset").addEventListener("change", loadTopicsView);

// ----------------------------- init -----------------------------
loadDatasets().catch(err => {
    document.getElementById("searchNote").className = "note err";
    document.getElementById("searchNote").textContent = "Cannot reach API: " + err.message;
});
