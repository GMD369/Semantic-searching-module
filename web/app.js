(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const queryInput = $("#q");
  const controls = $("#controls");
  const resultsEl = $("#results");
  const statusEl = $("#status");
  const errorEl = $("#error");
  const introEl = $("#intro");
  const template = $("#result-template");
  let meta = null;
  let requestId = 0;
  let debounce = null;
  let conceptK = null;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const fmt = (v, d = 2) => Number(v).toFixed(d);

  function renderSegments(target, segments) {
    target.replaceChildren();
    for (const s of segments) {
      if (!s.kind) { target.append(document.createTextNode(s.text)); continue; }
      const mark = el("mark", "m-" + s.kind, s.text);
      mark.title = `${s.kind === "exact" ? "Word you typed" : "Related word found by the model"} · adds ${fmt(s.value, 3)} to the score`;
      target.append(mark);
    }
  }

  function chip(text, weight, cls, title) {
    const c = el("span", "tchip " + (cls || ""), text);
    if (weight !== undefined && weight !== null) c.append(el("small", "", fmt(weight)));
    if (title) c.title = title;
    return c;
  }

  function renderStatus(data, compare) {
    statusEl.hidden = false;
    introEl.hidden = true;
    const qt = $("#query-terms");
    qt.replaceChildren();
    data.query_vector.forEach(t => qt.append(chip(t.word, t.weight, "", `TF-IDF weight = (1 + ln tf) × idf, idf = ${fmt(t.idf)}`)));
    data.unknown.forEach(w => qt.append(chip(w, null, "unknown", "Not in the collection's vocabulary, so it has no dimension")));

    const ex = $("#expanded");
    ex.replaceChildren();
    data.expanded_query.forEach(t => ex.append(chip(t.word, t.relative, t.typed ? "" : "learned",
      t.typed ? "Typed word, relative weight" : "Word the model added to your query, relative weight")));

    const row = $("#concept-row");
    const qc = $("#query-concepts");
    qc.replaceChildren();
    row.hidden = !data.query_concepts.length;
    data.query_concepts.forEach(c => {
      const span = el("span", "cchip" + (c.weight < 0 ? " neg" : ""));
      span.append(el("b", "", String(c.concept)), `${c.label.slice(0, 3).join(", ")} · ${fmt(Math.abs(c.weight))}`);
      span.title = `Concept ${c.concept}: share of the query vector pointing this way`;
      qc.append(span);
    });

    const counts = $("#counts");
    counts.replaceChildren();
    counts.append(`${data.model_label}. Showing `, el("strong", "", String(data.total_shown)),
      ` of ${data.total_positive} documents with similarity above 0 (of ${data.total_considered}).`);
    if (compare && data.model !== "vsm") {
      const meaning = data.results.filter(r => r.found_by_meaning).length;
      counts.append(" Exact-word VSM finds ", el("strong", "", String(data.vsm_zero_hits)), " documents.");
      if (meaning) counts.append(" ", el("strong", "", String(meaning)), " shown here share no word with your query.");
    }
    highlightConcepts(data.query_concepts.map(c => c.concept));
  }

  function contributionTable(table, words, other) {
    table.replaceChildren();
    table.innerHTML = "<thead><tr><th>Word</th><th>Source</th><th>Query weight</th><th>Doc weight</th><th>Adds</th></tr></thead>";
    const body = el("tbody");
    for (const w of words) {
      const tr = el("tr");
      const src = el("td");
      src.append(el("span", "src" + (w.typed ? "" : " learned"), w.typed ? "typed" : "related"));
      tr.append(el("td", "", w.word), src, el("td", "num", fmt(w.query_weight, 3)),
        el("td", "num", fmt(w.doc_weight, 3)), el("td", "num", "+" + fmt(w.contribution, 3)));
      body.append(tr);
    }
    if (Math.abs(other) >= 0.0005) {
      const tr = el("tr");
      tr.append(el("td", "", "all other words"), el("td"), el("td"), el("td"),
        el("td", "num", (other >= 0 ? "+" : "") + fmt(other, 3)));
      body.append(tr);
    }
    table.append(body);
  }

  function conceptBars(box, concepts, model) {
    box.replaceChildren();
    if (!concepts) {
      box.append(el("p", "muted-note", model === "vsm"
        ? "VSM has no concepts: each word is its own axis, so only typed words can score."
        : "GVSM has no concepts: it links words through how often they occur in the same documents."));
      return;
    }
    if (!concepts.length) { box.append(el("p", "muted-note", "No concept contributes clearly.")); return; }
    const peak = Math.max(...concepts.map(c => c.contribution));
    for (const c of concepts) {
      const row = el("div", "bar-row");
      const label = el("span", "lbl", `${c.concept} · ${c.label.join(", ")}`);
      label.title = label.textContent;
      const track = el("div", "track");
      const fill = el("div", "fill");
      fill.style.width = (100 * c.contribution / peak).toFixed(1) + "%";
      track.append(fill);
      row.append(label, el("span", "num", "+" + fmt(c.contribution, 3)), track);
      box.append(row);
    }
  }

  function renderResults(data, compare) {
    resultsEl.replaceChildren();
    if (!data.results.length) {
      resultsEl.append(el("div", "empty", "No document reaches the minimum similarity. Lower the slider or describe the problem with other words."));
      $("#matrix-panel").hidden = true;
      return;
    }
    for (const r of data.results) {
      const card = template.content.firstElementChild.cloneNode(true);
      const doc = r.document;
      const pct = Math.max(0, Math.min(1, r.score)) * 100;
      const ring = $(".ring", card);
      ring.style.setProperty("--p", pct.toFixed(1));
      ring.setAttribute("aria-label", `Cosine similarity ${fmt(r.score)}`);
      $(".value", card).textContent = fmt(r.score);
      $(".rank", card).textContent = "#" + r.rank;
      $(".type", card).textContent = doc.type;
      const flag = $(".vsm-flag", card);
      if (compare && data.model !== "vsm") {
        if (r.found_by_meaning) {
          flag.textContent = "Found by meaning · shares no word with your query";
          flag.classList.add("meaning");
          card.classList.add("is-meaning");
        } else {
          flag.textContent = `Exact-word VSM score ${fmt(r.vsm_score)}`;
        }
      } else {
        flag.remove();
      }
      renderSegments($(".title", card), r.highlights.title);
      $(".meta", card).textContent = [doc.id, doc.factory_name, doc.zone, doc.meter, doc.severity, doc.date]
        .filter(Boolean).join(" · ");
      renderSegments($(".snippet", card), r.highlights.text);
      $(".formula-line", card).textContent = r.explanation.formula;
      contributionTable($(".contrib", card), r.explanation.words, r.explanation.other_words);
      conceptBars($(".concept-bars", card), r.explanation.concepts, data.model);
      if (r.rank === 1) $(".why", card).open = true;
      resultsEl.append(card);
    }
    renderMatrix(data.matrix);
  }

  function renderMatrix(m) {
    const panel = $("#matrix-panel");
    const table = $("#matrix");
    panel.hidden = !m.docs.length;
    table.replaceChildren();
    const head = el("thead");
    const hr = el("tr");
    hr.append(el("th", "", "term"));
    m.docs.forEach(d => hr.append(el("th", "", d)));
    hr.append(el("th", "qcol", "q (typed)"), el("th", "", "q read as"));
    head.append(hr);
    const body = el("tbody");
    m.terms.forEach((t, i) => {
      const tr = el("tr");
      const name = el("td", "", t.word);
      if (!t.typed) name.append(" ", el("span", "tchip learned", "related"));
      tr.append(name);
      m.values[i].forEach(v => tr.append(el("td", v ? "" : "zero", v ? fmt(v, 3) : "0")));
      tr.append(el("td", "qcol" + (m.query[i] ? "" : " zero"), fmt(m.query[i], 3)), el("td", "", fmt(m.expanded[i], 3)));
      body.append(tr);
    });
    table.append(head, body);
  }

  async function loadConcepts(k) {
    if (k === conceptK) return;
    conceptK = k;
    try {
      const data = await (await fetch("/api/concepts?k=" + k)).json();
      if (conceptK !== k) return;
      const list = $("#concepts");
      list.replaceChildren();
      $("#concept-k").textContent = `(k = ${data.k})`;
      for (const c of data.concepts) {
        const li = el("li");
        li.dataset.concept = c.concept;
        li.append(el("b", "", String(c.concept)), el("span", "", c.positive.join(", ")));
        li.title = `σ = ${c.singular_value} · ${(c.energy * 100).toFixed(1)}% of Σσ² · strongest documents ${c.documents.join(", ")}`;
        list.append(li);
      }
    } catch (error) { /* the search error message covers a dead server */ }
  }

  function highlightConcepts(active) {
    document.querySelectorAll("#concepts li").forEach(li =>
      li.classList.toggle("active", active.includes(Number(li.dataset.concept))));
  }

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = !message;
  }

  function params() {
    const form = new FormData(controls);
    const p = new URLSearchParams();
    p.set("q", queryInput.value);
    p.set("model", form.get("model"));
    p.set("k", form.get("k"));
    p.set("top", form.get("top"));
    p.set("min", form.get("min"));
    return p;
  }

  function updateK() {
    const k = Number($("#k").value);
    $("#k-value").textContent = k;
    if (meta) $("#energy").textContent = Math.round(meta.energy[k - 1] * 100) + "%";
    loadConcepts(k);
  }

  async function runSearch() {
    const url = new URL(location.href);
    queryInput.value.trim() ? url.searchParams.set("q", queryInput.value) : url.searchParams.delete("q");
    history.replaceState(null, "", url);
    if (!queryInput.value.trim()) {
      resultsEl.replaceChildren();
      statusEl.hidden = true;
      introEl.hidden = false;
      $("#matrix-panel").hidden = true;
      highlightConcepts([]);
      showError("");
      return;
    }
    const id = ++requestId;
    try {
      const response = await fetch("/api/search?" + params().toString());
      const data = await response.json();
      if (id !== requestId) return;
      if (!response.ok) {
        showError(data.error || "Search failed.");
        resultsEl.replaceChildren();
        statusEl.hidden = true;
        $("#matrix-panel").hidden = true;
        return;
      }
      showError("");
      const compare = new FormData(controls).get("compare") === "on";
      renderStatus(data, compare);
      renderResults(data, compare);
    } catch (error) {
      if (id === requestId) showError("Cannot reach the search server. Is python app.py still running?");
    }
  }

  function schedule(delay = 220) {
    clearTimeout(debounce);
    debounce = setTimeout(runSearch, delay);
  }

  async function init() {
    try {
      meta = await (await fetch("/api/meta")).json();
      const k = $("#k");
      k.max = meta.max_k;
      k.value = meta.default_k;
      const examples = $("#examples");
      for (const example of meta.examples) {
        const button = el("button", "example");
        button.type = "button";
        button.append(el("code", "", example.query), el("small", "", example.note));
        button.addEventListener("click", () => { queryInput.value = example.query; schedule(0); queryInput.focus(); });
        examples.append(button);
      }
    } catch (error) {
      showError("Cannot reach the search server. Start it with: python app.py");
    }
    updateK();

    queryInput.addEventListener("input", () => schedule());
    $("#search-form").addEventListener("submit", event => { event.preventDefault(); schedule(0); });
    controls.addEventListener("change", () => schedule(0));
    $("#k").addEventListener("input", () => { updateK(); schedule(150); });
    $("#min").addEventListener("input", event => {
      $("#min-value").textContent = fmt(event.target.value);
      schedule(120);
    });
    $("#reset").addEventListener("click", () => {
      controls.reset();
      if (meta) $("#k").value = meta.default_k;
      $("#min-value").textContent = "0.10";
      updateK();
      schedule(0);
    });
    document.addEventListener("keydown", event => {
      const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
      if (event.key === "/" && !typing) { event.preventDefault(); queryInput.focus(); }
    });

    const initial = new URL(location.href).searchParams;
    if (initial.get("model")) {
      const radio = controls.querySelector(`input[name=model][value="${CSS.escape(initial.get("model"))}"]`);
      if (radio) radio.checked = true;
    }
    if (initial.has("matrix")) $("#matrix-panel").open = true;
    if (initial.get("q")) {
      queryInput.value = initial.get("q");
      runSearch();
    }
  }

  init();
})();
