/* Question Studio – shared shell: theme, sidebar, icons and result renderers */
(function () {
  "use strict";
  document.documentElement.dataset.theme = localStorage.getItem("qs_theme") || "light";

  const LV = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];
  const ICONS = {
    mark: '<rect x="8" y="4" width="8" height="4" rx="1" fill="currentColor" stroke="none"/><rect x="5" y="10" width="14" height="4" rx="1" fill="currentColor" stroke="none"/><rect x="2" y="16" width="20" height="4" rx="1" fill="currentColor" stroke="none"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    tools: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    up: '<path d="M12 19V5M5 12l7-7 7 7"/>',
    down: '<path d="M12 5v14M19 12l-7 7-7-7"/>',
    pencil: '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    chart: '<path d="M12 20V10M18 20V4M6 20v-4"/>',
    spark: '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 17v4M17 19h4"/>',
    send: '<path d="M22 2L11 13M22 2l-7 20-4-9-9-4z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    menu: '<path d="M3 12h18M3 6h18M3 18h18"/>',
    back: '<path d="M19 12H5M12 19l-7-7 7-7"/>',
    warn: '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0zM12 9v4M12 17h.01"/>',
    check: '<path d="M22 11.1V12a10 10 0 1 1-5.9-9.1M22 4L12 14l-3-3"/>',
    clip: '<path d="M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M9 13h6M9 17h6"/>'
  };
  const icon = (n, s = 16) => `<svg class="ic" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[n] || ""}</svg>`;
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = s => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");

  /* ---------- result renderers (used by the assistant and the classic page) ---------- */
  const metric = (k, v) => `<div class="metric"><span>${esc(k)}</span><div class="track"><i style="width:${v}%"></i></div><b>${v}</b></div>`;
  const scale = (cur, nxt) => `<div class="scale">${LV.map((l, i) => `<div class="step ${l === cur ? "cur" : l === nxt ? "nxt" : ""}"><i>${i + 1}</i>${l}</div>`).join("")}</div>`;
  const lvBadge = p => `<span class="badge lv">${esc(p.level)}</span><span class="badge d-${String(p.difficulty).toLowerCase()}">${esc(p.difficulty)}</span>`;
  const refs = r => (r || []).map(x => `<blockquote><b>AIMA p.${esc(x.page)}</b> — ${esc(x.text.slice(0, 260))}…</blockquote>`).join("");
  const tone = g => ({ Excellent: "ok", Good: "ok", "Needs work": "warn", Weak: "bad" }[g] || "");
  const sbtn = (ic, label, text, shown) => `<button class="btn sm" data-send="${esc(text)}" data-label="${esc(shown || label)}">${icon(ic, 14)}${esc(label)}</button>`;
  const conf = p => `<div class="muted sm" style="margin-top:6px">${Math.round(p.level_confidence * 100)}% confidence</div>`;
  const warnBox = t => `<div class="callout warn"><div class="head">${icon("warn", 15)}Notice</div><p>${esc(t)}</p></div>`;

  const R = {
    evaluation: r => { const f = r.relevance || { status: "not_evaluated", reason: "No evaluation was returned.", explains: "" }, ok = f.status === "ok", ev = f.evidence, src = r.faithfulness;
      const chips = (arr, cls) => (arr || []).map(w => `<span class="badge ${cls}">${esc(w)}</span>`).join("") || `<span class="muted sm">none</span>`;
      const bandCls = { Strong: "d-easy", Moderate: "d-medium", Weak: "d-hard" }[f.band] || "";
      return `<section class="card">
      <div class="split">
        <div><div class="label">Keyword relevance</div>${ok ? `<div class="score">${f.score}<small>%</small> <span class="badge ${bandCls}" style="vertical-align:middle">${esc(f.band)}</span></div>` : `<div class="score na">Not evaluated</div>`}
          <div class="muted sm" style="margin-top:6px;max-width:34em">${esc(ok ? f.method : f.reason || "No reliable evaluation was possible.")}</div></div>
        <div><div class="label">Predicted level</div>${lvBadge(r.prediction)}${conf(r.prediction)}</div>
      </div>
      <div class="muted sm" style="margin-top:12px">${esc(f.explains)}</div>
      ${f.note ? warnBox(f.note) : ""}
      ${ev && ev.kept ? `<details open><summary>How the score was calculated</summary>
        <div class="qm"><b class="sm">1. Common words removed:</b> ${chips(ev.removed, "")}</div>
        <div class="qm"><b class="sm">2. Keywords kept:</b> ${ok ? (ev.terms || []).map(t => `<span class="badge ${t.support >= 0.65 ? "d-easy" : t.support > 0 ? "d-medium" : "d-hard"}" title="${esc(t.why)}">${esc(t.term)}</span>`).join("") : chips(ev.kept, "")}</div>
        ${ok ? `<div class="qm"><b class="sm">3. Best matching topic:</b> ${esc(ev.topic)} <span class="muted sm">(${esc(ev.source)})</span></div>
        <div class="qm"><b class="sm">4. Focus on one topic:</b> ${Math.round(ev.focus * 100)}% <span class="muted sm">· green = strongly tied to the topic, amber = general word, red = not in the syllabus</span></div>` : ""}</details>` : ""}
      ${scale(r.prediction.level, r.prediction.next_level)}
      ${r.suggestions.length ? `<div class="label mt">Writing tips</div><ul class="clean">${r.suggestions.map(s => `<li>${esc(s)}</li>`).join("")}</ul>` : ""}
      ${r.ai_review ? `<div class="callout ok"><div class="head">AI review</div><p>${esc(r.ai_review.verdict)}</p><div class="rewrite">${esc(r.ai_review.improved)}</div></div>` : ""}
      ${src && src.status === "ok" && src.evidence ? `<details><summary>Supporting check: source text (${src.score}% of the terms found together)</summary>
        <div class="muted sm" style="margin:6px 0">Source: ${esc(src.evidence.source)}</div>${src.evidence.passage ? `<blockquote>${esc(src.evidence.passage)}…</blockquote>` : ""}</details>` : ""}
      <details><summary>Textbook references</summary>${refs(r.textbook_refs)}</details></section>`; },

    prediction: p => `<section class="card"><div class="label">Predicted level</div>${lvBadge(p)}${conf(p)}${scale(p.level, p.next_level)}
      <div class="metrics">${LV.map(l => metric(l, Math.round((p.level_probs[l] || 0) * 100))).join("")}</div></section>`,

    upgrade: u => u.upgraded ? `<section class="card"><div class="label">${u.direction === "down" ? "Degrade" : "Upgrade"} · ${esc(u.prediction.level)} to ${esc(u.target_level)}</div>
      <div class="rewrite">${esc(u.upgraded)}</div>
      <div class="muted sm" style="margin-top:8px">${esc(u.method)}${u.verified_level ? ` · model re-check: ${esc(u.verified_level)}` : ""}${u.why ? ` · ${esc(u.why)}` : ""}</div>
      ${u.verbs_for_target ? `<div class="muted sm" style="margin-top:6px">Typical ${esc(u.target_level)} verbs: ${u.verbs_for_target.map(esc).join(", ")}</div>` : ""}
      <details><summary>Real ${esc(u.target_level)} questions from the question bank</summary>${(u.examples || []).map(x => `<div class="mini-q">${esc(x.question)} <span class="muted">· ${esc(x.level)} · ${esc(x.difficulty)}</span></div>`).join("")}</details></section>`
      : `<section class="card"><div class="label">Predicted level</div>${lvBadge(u.prediction)}${scale(u.prediction.level, u.prediction.next_level)}<p style="margin:10px 0 0;color:var(--text-2)">${esc(u.message || "")}</p></section>`,

    questions: (r, o = { actions: true }) => `<section class="card">
      <div class="label">Generated questions · source: ${esc(r.source)}${r.filtered_out ? ` · ${r.filtered_out} off-style filtered out` : ""}</div>
      ${r.note ? warnBox(r.note) : ""}
      ${r.questions.map((x, i) => `<div class="qrow"><div class="qn">${i + 1}</div><div style="min-width:0">
        <div class="qt">${esc(x.question)}</div><div class="qm">${lvBadge(x)}</div>
        ${o.actions !== false ? `<div class="acts">${sbtn("chart", "Evaluate", "/evaluate " + x.question, "Evaluate question " + (i + 1))}${sbtn("up", "Upgrade", "/upgrade " + x.question, "Upgrade question " + (i + 1))}${sbtn("down", "Degrade", "/degrade " + x.question, "Degrade question " + (i + 1))}</div>` : ""}
        </div></div>`).join("")}</section>`
  };

  R.paper = (d, o = { actions: true }) => {
    const S = d.summary, li = l => LV.indexOf(l) + 1, act = o.actions !== false;
    const dot = l => `<span class="lvdot" style="background:var(--l${li(l)})"></span>`;
    const hasTag = d.questions.some(q => q.declared_level), hasMarks = d.questions.some(q => q.marks != null);
    return `<section class="card" data-paper="${esc(JSON.stringify(d))}">
      <div class="split"><div style="min-width:0"><div class="label">Question paper analysis</div>
        <div class="pname">${icon("file", 16)}${esc(d.filename)}</div>
        <div class="muted sm">${S.total} question${S.total === 1 ? "" : "s"} · ${d.pages} page${d.pages === 1 ? "" : "s"}${S.total_marks ? ` · ${S.total_marks} marks detected` : ""}</div></div>
        <div class="acts" style="margin:0;align-self:flex-start"><button class="btn sm" data-export="xlsx">${icon("download", 14)}Excel</button><button class="btn sm" data-export="csv">${icon("download", 14)}CSV</button></div></div>
      <div class="stack" role="img" aria-label="Bloom level distribution">${S.by_level.filter(x => x.count).map(x => `<i style="flex:${x.count};background:var(--l${li(x.level)})" title="${x.level}: ${x.count}"></i>`).join("")}</div>
      <div class="lstats">${S.by_level.map(x => `<div class="lstat"><div class="lh">${dot(x.level)}${x.level}</div><div class="lc"><b>${x.count}</b><em>${x.percent}%</em></div></div>`).join("")}</div>
      ${S.insights.length ? `<div class="label mt">Insights</div><ul class="clean">${S.insights.map(t => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}
      <div class="label mt">Questions</div>
      <div class="tbl-wrap"><table class="tbl"><thead><tr><th>No.</th><th>Question</th>${hasMarks ? "<th>Marks</th>" : ""}<th>Bloom level</th><th>Difficulty</th>${hasTag ? "<th>Paper tag</th>" : ""}</tr></thead><tbody>
      ${d.questions.map(q => `<tr><td class="c-n">${esc(q.id)}</td>
        <td class="c-q">${q.context ? `<div class="ctx">${esc(q.context.slice(0, 140))}${q.context.length > 140 ? "…" : ""}</div>` : ""}${esc(q.text)}${q.type === "MCQ" ? ` <span class="badge">MCQ · ${q.options} options</span>` : ""}
          ${act ? `<div class="acts">${sbtn("up", "Upgrade", "/upgrade " + q.text, "Upgrade question " + q.id)}${sbtn("down", "Degrade", "/degrade " + q.text, "Degrade question " + q.id)}</div>` : ""}</td>
        ${hasMarks ? `<td>${q.marks ?? "–"}</td>` : ""}
        <td style="white-space:nowrap">${dot(q.level)}<b>${esc(q.level)}</b> <span class="muted sm">${Math.round(q.level_confidence * 100)}%</span>
          ${q.low_confidence ? `<div class="sm warn-t">Low confidence${q.alt_level ? ` · or ${esc(q.alt_level)}` : ""}</div>` : q.alt_level ? `<div class="muted sm">or ${esc(q.alt_level)} (${Math.round(q.alt_confidence * 100)}%)</div>` : ""}</td>
        <td><span class="badge d-${String(q.difficulty).toLowerCase()}">${esc(q.difficulty)}</span></td>
        ${hasTag ? `<td>${q.declared_level ? (q.declared_level === q.level ? `<span class="ok-t">${esc(q.declared_level)}</span>` : `<span class="warn-t">${esc(q.declared_level)} (differs)</span>`) : "–"}</td>` : ""}
</tr>`).join("")}
      </tbody></table></div>
      ${d.ocr ? `<div class="callout warn"><div class="head">${icon("warn", 15)}Read with OCR</div><p>${esc(d.ocr.engine)} · ${Math.round(d.ocr.confidence * 100)}% confidence. Scanned text can contain mistakes, so check each question.</p></div>` : ""}
      ${(d.notes || []).filter(n => !/^Read with OCR/.test(n)).map(n => `<div class="muted sm" style="margin-top:8px">${esc(n)}</div>`).join("")}
      ${d.raw_text ? `<details><summary>Show the text I read from the file</summary><pre class="raw">${esc(d.raw_text)}</pre></details>` : ""}</section>`;
  };

  R.doc_questions = (d, o = { actions: true }) => {
    const li = l => LV.indexOf(l) + 1, act = o.actions !== false, built = d.kind === "paper_build";
    const topics = built ? new Set(d.questions.map(q => q.section)).size : 0;
    let prev = null;
    const row = q => {
      const head = built && q.section !== prev ? (prev = q.section, `<div class="sec">${esc(q.section)}</div>`) : "";
      return head + `<div class="qrow"><div class="qn">${q.no}</div><div style="min-width:0;flex:1">
        <div class="qt">${esc(q.question)}</div>
        ${q.options ? `<ul class="opts">${q.options.map((o, i) => `<li>${"ABCD"[i]}) ${esc(o)}</li>`).join("")}</ul>` : ""}
        <div class="qm"><span class="badge lv"><span class="lvdot" style="background:var(--l${li(q.level)});margin-right:6px"></span>${esc(q.level)}</span><span class="badge d-${String(q.difficulty).toLowerCase()}">${esc(q.difficulty)}</span>${built ? "" : `<span class="badge">${esc(q.type)}</span>`}<span class="badge">${q.marks} mark${q.marks === 1 ? "" : "s"}</span>${q.source && !built ? `<span class="badge">${esc(q.source)}</span>` : ""}</div>
        ${q.answer || q.reference ? `<details><summary>${q.answer ? "Model answer" : "Reference from your document"}</summary><p style="margin:6px 0 0;color:var(--text-2)">${esc(q.answer || q.reference)}</p></details>` : ""}
        ${act ? `<div class="acts">${sbtn("up", "Upgrade", "/upgrade " + q.question, "Upgrade question " + q.no)}${sbtn("down", "Degrade", "/degrade " + q.question, "Degrade question " + q.no)}</div>` : ""}
        </div></div>`;
    };
    return `<section class="card" data-qset="${esc(JSON.stringify(d))}">
      <div class="split"><div style="min-width:0"><div class="label">${built ? "Question paper" : "Questions from your document"}</div>
        <div class="pname">${icon("file", 16)}${built ? "Question paper" : esc(d.filename)}</div>
        <div class="muted sm">${d.questions.length} question${d.questions.length === 1 ? "" : "s"} · ${d.total_marks} marks${built ? ` · ${topics} topic${topics === 1 ? "" : "s"}` : ` · ${esc(d.spec.qtype)}${d.source === "offline" ? " · written from key terms" : ""}`}</div></div>
        <div class="acts" style="margin:0;align-self:flex-start"><button class="btn sm" data-export="docx">${icon("download", 14)}Word</button><button class="btn sm" data-export="xlsx">${icon("download", 14)}Excel</button></div></div>
      ${built && d.blueprint ? `<details open><summary>Blueprint</summary><div class="bp">${d.blueprint.map(b => `<span class="badge">${esc(b.topic)}: ${Object.entries(b.counts).filter(([, n]) => n).map(([l, n]) => n + " " + l).join(", ")}</span>`).join("")}</div></details>` : ""}
      ${(d.notes || []).map(n => `<div class="callout"><p style="margin:0">${esc(n)}</p></div>`).join("")}
      ${d.questions.map(row).join("")}</section>`;
  };

  /* ---------- question-paper builder: pick topics, set questions per Bloom level ---------- */
  const ABBR = { Remember: "Rem", Understand: "Und", Apply: "App", Analyze: "Ana", Evaluate: "Eva", Create: "Cre" };
  R.paper_plan = d => {
    const on = id => (d.preselected || []).includes(id);
    const pill = (id, label, title, cls = "") => `<button type="button" class="pill${on(id) ? " on" : ""}${cls}" data-tid="${esc(id)}" data-title="${esc(title)}" data-q="${esc((label + " " + title).toLowerCase())}">${esc(label)}</button>`;
    const chapter = c => {
      const head = pill(c.id, `${c.number}. ${c.title}`, `${c.number} ${c.title}`, " pill-ch");
      return c.sections.length
        ? `<details class="chap"><summary>${head}<span class="muted sm">${c.sections.length} subtopic${c.sections.length === 1 ? "" : "s"}</span></summary><div class="pills sub">${c.sections.map(x => pill(x.id, `${x.number} ${x.title}`, `${x.number} ${x.title}`)).join("")}</div></details>`
        : `<div class="chap">${head}</div>`;
    };
    const units = d.units.map(u => `<details class="unit" open><summary class="unit-h">${esc(u.name)} <span class="muted sm">· ${u.chapters.length} chapter${u.chapters.length === 1 ? "" : "s"}</span></summary>${u.chapters.map(chapter).join("")}</details>`).join("");
    const docs = d.doc_topics && d.doc_topics.length ? `<details class="unit" open><summary class="unit-h">From your document (${esc(d.doc)}) <span class="muted sm">· ${d.doc_topics.length} topic${d.doc_topics.length === 1 ? "" : "s"}</span></summary><div class="pills sub">${d.doc_topics.map(t => pill(t.id, t.title, t.title)).join("")}</div></details>` : "";
    const total = `${d.counts.chapters} chapters and ${d.counts.sections} subtopics` + (d.doc_topics && d.doc_topics.length ? ` + ${d.doc_topics.length} from your document` : "");
    return `<section class="card planner" data-planner="${esc(JSON.stringify({ levels: d.levels, limits: d.limits }))}" data-total="${esc(total)}">
      <div class="label">Question paper builder</div>
      <div class="plabel">1. Choose topics <span class="muted sm">(tap a chapter or any subtopic; up to ${d.limits.topics})</span></div>
      <div class="plan-tools"><input type="search" class="input plan-filter" placeholder="Filter topics, e.g. decision, search, Bayesian…" aria-label="Filter topics"><button type="button" class="btn sm plan-expand">Expand all</button><button type="button" class="btn sm plan-collapse">Collapse all</button></div>
      <div class="muted sm plan-count">${esc(total)} available</div>
      <div class="plan-topics">${docs}${units}</div>
      <div class="plabel">2. How many questions at each level?</div>
      <div class="tbl-wrap"><table class="tbl plan"><thead><tr><th>Topic</th>${d.levels.map(l => `<th class="c" title="${l}">${ABBR[l]}</th>`).join("")}<th class="c">Total</th></tr></thead>
        <tbody class="plan-rows"></tbody><tfoot><tr><td><b>All topics</b></td><td colspan="${d.levels.length}"></td><td class="c"><b class="plan-total">0</b></td></tr></tfoot></table></div>
      <div class="plan-bar"><label class="sm muted">Marks per question (optional)<input type="number" class="input plan-marks" min="1" max="50" placeholder="automatic"></label>
        <button type="button" class="btn primary plan-go">${icon("spark", 14)}Generate question paper</button></div>
      <div class="muted sm plan-hint">Select at least one topic above.</div></section>`;
  };
  const DEFAULT_ROW = [1, 1, 1, 1, 0, 0];
  const planTotals = card => {
    let all = 0;
    card.querySelectorAll(".plan-rows tr").forEach(tr => { const t = [...tr.querySelectorAll(".cnt")].reduce((a, i) => a + (+i.value || 0), 0); tr.querySelector(".rowtotal").textContent = t; all += t; });
    card.querySelector(".plan-total").textContent = all;
    const lim = JSON.parse(card.dataset.planner).limits, hint = card.querySelector(".plan-hint"), n = card.querySelectorAll(".plan-rows tr").length;
    hint.textContent = !n ? "Select at least one topic above." : all > lim.total ? `That is ${all} questions; a paper can have at most ${lim.total}.` : all === 0 ? "Enter how many questions you want at each level." : `${all} question${all === 1 ? "" : "s"} across ${n} topic${n === 1 ? "" : "s"}.`;
    hint.classList.toggle("warn-t", all > lim.total);
    return all;
  };
  const addPlanRow = (card, pill) => {
    const lv = JSON.parse(card.dataset.planner).levels;
    card.querySelector(".plan-rows").insertAdjacentHTML("beforeend", `<tr data-tid="${esc(pill.dataset.tid)}"><td class="c-q">${esc(pill.dataset.title)}</td>${lv.map((l, i) => `<td class="c"><input type="number" class="cnt" data-level="${l}" min="0" max="${JSON.parse(card.dataset.planner).limits.cell}" value="${DEFAULT_ROW[i] || 0}" aria-label="${esc(pill.dataset.title)} ${l}"></td>`).join("")}<td class="c rowtotal">0</td></tr>`);
  };
  document.addEventListener("click", e => {
    const pill = e.target.closest(".planner .pill");
    if (pill) {
      e.preventDefault();
      const card = pill.closest(".planner"), on = pill.classList.contains("on");
      if (!on && card.querySelectorAll(".pill.on").length >= JSON.parse(card.dataset.planner).limits.topics) { card.querySelector(".plan-hint").textContent = "You can choose at most " + JSON.parse(card.dataset.planner).limits.topics + " topics."; return; }
      pill.classList.toggle("on", !on);
      if (on) card.querySelector(`.plan-rows tr[data-tid="${CSS.escape(pill.dataset.tid)}"]`)?.remove(); else addPlanRow(card, pill);
      planTotals(card);
      return;
    }
    const go = e.target.closest(".plan-go");
    if (go) {
      const card = go.closest(".planner"), lim = JSON.parse(card.dataset.planner).limits, total = planTotals(card);
      const topics = [...card.querySelectorAll(".plan-rows tr")].map(tr => ({ id: tr.dataset.tid, counts: Object.fromEntries([...tr.querySelectorAll(".cnt")].map(i => [i.dataset.level, Math.max(0, Math.min(lim.cell, +i.value || 0))]).filter(([, n]) => n > 0)) })).filter(t => Object.keys(t.counts).length);
      if (!topics.length || total === 0 || total > lim.total) { card.querySelector(".plan-hint").textContent = !topics.length ? "Select a topic and enter at least one question." : `Please keep the total between 1 and ${lim.total}.`; return; }
      const marks = card.querySelector(".plan-marks").value;
      card.querySelectorAll("button,input").forEach(x => x.disabled = true);
      document.dispatchEvent(new CustomEvent("qs:prompt", { detail: { text: "/paperplan " + JSON.stringify({ topics, marks: marks ? +marks : null }), label: `Generate question paper (${total} questions, ${topics.length} topic${topics.length === 1 ? "" : "s"})` } }));
    }
  });
  document.addEventListener("input", e => { const i = e.target.closest(".planner .cnt"); if (i) { i.value = Math.max(0, Math.min(+i.max || 10, Math.round(+i.value || 0))); planTotals(i.closest(".planner")); } });
  /* filter + expand / collapse for the long topic list: nothing is removed, matching topics are simply brought into view */
  const planFilter = card => {
    const q = card.querySelector(".plan-filter").value.trim().toLowerCase(), pills = [...card.querySelectorAll(".plan-topics .pill")];
    pills.forEach(p => p.classList.toggle("flt-hide", !!q && !p.dataset.q.includes(q) && !p.classList.contains("on")));
    const visible = el => [...el.querySelectorAll(".pill")].some(p => !p.classList.contains("flt-hide"));
    card.querySelectorAll(".plan-topics details.chap").forEach(dt => { const any = visible(dt); dt.classList.toggle("flt-hide", !!q && !any); if (q) dt.open = any; else dt.open = !!dt.querySelector(".pill.on"); });
    card.querySelectorAll(".plan-topics details.unit").forEach(u => { const any = visible(u); u.classList.toggle("flt-hide", !!q && !any); if (q && any) u.open = true; });
    card.querySelector(".plan-count").textContent = q ? `${pills.filter(p => !p.classList.contains("flt-hide")).length} matching topic(s)` : card.dataset.total + " available";
  };
  document.addEventListener("input", e => { const f = e.target.closest(".planner .plan-filter"); if (f) planFilter(f.closest(".planner")); });
  document.addEventListener("click", e => {
    const ex = e.target.closest(".planner .plan-expand, .planner .plan-collapse");
    if (ex) ex.closest(".planner").querySelectorAll(".plan-topics details").forEach(dt => { dt.open = ex.classList.contains("plan-expand"); });
  });

  const saveBlob = (blob, name) => { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500); };
  document.addEventListener("click", async e => {
    const b = e.target.closest("[data-export]");
    if (!b) return;
    const card = b.closest("[data-paper],[data-qset]"), isSet = !!card.dataset.qset;
    const d = JSON.parse(isSet ? card.dataset.qset : card.dataset.paper), base = (d.filename || "export").replace(/\.[a-z0-9]+$/i, ""), old = b.innerHTML, kind = b.dataset.export;
    if (kind === "csv") {
      const cell = v => { let t = String(v ?? ""); if (/^[=+\-@]/.test(t)) t = "'" + t; return '"' + t.replace(/"/g, '""') + '"'; };
      const rows = [["No.", "Section", "Question", "Marks", "Bloom level (predicted)", "Difficulty", "Confidence", "Paper's own tag"]]
        .concat(d.questions.map(q => [q.id, q.section, q.text, q.marks, q.level, q.difficulty, Math.round(q.level_confidence * 100) + "%", q.declared_level]));
      saveBlob(new Blob(["\ufeff" + rows.map(r => r.map(cell).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" }), base + "_bloom_analysis.csv");
      return;
    }
    b.disabled = true; b.textContent = "Preparing…";
    try {
      const url = isSet ? "/api/export-questions?fmt=" + kind : "/api/export-paper";
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(d) });
      if (!r.ok) throw new Error();
      saveBlob(await r.blob(), isSet ? base + (kind === "docx" ? "_question_paper.docx" : "_questions.xlsx") : base + "_bloom_analysis.xlsx");
    } catch (_) { b.textContent = "Export failed"; await new Promise(r => setTimeout(r, 1800)); }
    b.innerHTML = old; b.disabled = false;
  });

  /* ---------- shell: sidebar, theme, status ---------- */
  const PROMPTS = [
    ["Generate questions", "Generate questions"],
    ["Create a question paper", "Create a question paper"],
    ["What topics can I use?", "What topics can I ask about?"],
    ["Predict a question's level", "What level is: Define what makes a heuristic function admissible in A* search."],
    ["Upgrade a question", "Upgrade: Explain the PEAS framework for intelligent agents."],
    ["Degrade a question", "Degrade: Design a minimax procedure with alpha-beta pruning for a two-player game."]
  ];

  function build() {
    const page = document.body.dataset.page || "chat";
    const side = document.getElementById("side");
    if (side) {
      side.innerHTML = `
        <div class="brand"><span class="mark">${icon("mark", 18)}</span><div><b>Question Studio</b><small>Bloom's taxonomy · AIMA 3e</small></div></div>
        <button class="btn primary block" id="newchat">${icon("plus", 15)}New conversation</button>
        <nav class="nav">
          <a class="nav-item ${page === "chat" ? "active" : ""}" href="/">${icon("chat", 17)}Assistant</a>
          <a class="nav-item ${page === "classic" ? "active" : ""}" href="/classic">${icon("tools", 17)}Classic tools</a>
        </nav>
        <div class="side-label">Try asking</div>
        ${PROMPTS.map(p => `<button class="prompt" data-prompt="${esc(p[1])}">${esc(p[0])}</button>`).join("")}
        <div class="grow"></div>
        <div class="side-foot"><span>Appearance</span>
          <div class="seg" id="theme"><button data-t="light">${icon("sun", 14)}Light</button><button data-t="dark">${icon("moon", 14)}Dark</button></div></div>`;
    }
    document.querySelectorAll("[data-icon]").forEach(el => el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon, +el.dataset.size || 16)));

    const mark = () => document.querySelectorAll("#theme button").forEach(b => b.classList.toggle("on", b.dataset.t === document.documentElement.dataset.theme));
    mark();
    const close = () => { side && side.classList.remove("open"); const s = document.getElementById("scrim"); s && s.classList.remove("on"); };
    document.addEventListener("click", e => {
      const t = e.target.closest("#theme button");
      if (t) { document.documentElement.dataset.theme = t.dataset.t; localStorage.setItem("qs_theme", t.dataset.t); mark(); return; }
      if (e.target.closest("#menu")) { side.classList.add("open"); document.getElementById("scrim").classList.add("on"); return; }
      if (e.target.id === "scrim" || e.target.closest(".nav-item")) close();
      const p = e.target.closest("[data-prompt]");
      if (p) { close(); page === "chat" ? document.dispatchEvent(new CustomEvent("qs:prompt", { detail: p.dataset.prompt })) : (location.href = "/?q=" + encodeURIComponent(p.dataset.prompt)); }
      if (e.target.closest("#newchat")) { close(); page === "chat" ? document.dispatchEvent(new Event("qs:newchat")) : (location.href = "/?new=1"); }
    });

    fetch("/api/health").then(r => r.json()).then(h => {
      document.querySelectorAll(".js-status").forEach(el => { el.classList.toggle("on", !!h.llm); el.querySelector("span").textContent = h.llm ? "AI mode on" : "Offline mode"; });
    }).catch(() => document.querySelectorAll(".js-status span").forEach(s => s.textContent = "Server offline"));
  }

  window.QS = { LV, icon, esc, fmt, R, initPlanners: root => (root || document).querySelectorAll(".planner").forEach(card => {
    if (card.dataset.ready) return;
    card.dataset.ready = "1";
    card.querySelectorAll(".pill.on").forEach(p => addPlanRow(card, p));
    planTotals(card);
  }) };
  document.addEventListener("DOMContentLoaded", build);
})();
