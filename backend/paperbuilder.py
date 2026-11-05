"""Question-paper builder: pick several topics and say how many questions you want at each Bloom level.

Topics come from the syllabus (textbook chapters) and, when a document is uploaded, from that document's own headings.
Everything built here is inside the syllabus or the uploaded material by construction.
"""
import re

import engine, docgen, syllabus, guard

LEVELS = engine.LEVELS
MAX_TOTAL, MAX_TOPICS, MAX_CELL = 40, 8, 10
PAPER_MARKS = {"Remember": 2, "Understand": 3, "Apply": 5, "Analyze": 6, "Evaluate": 8, "Create": 10}
LV = r"remember|understand|apply|analy[sz]e|evaluate|create"
LV_MAP = {"remember": "Remember", "understand": "Understand", "apply": "Apply", "analyze": "Analyze", "analyse": "Analyze", "evaluate": "Evaluate", "create": "Create"}
DEFAULT_LEVELS = ["Understand", "Apply", "Analyze"]


# ----------------------------------------------------------------- topics
def doc_topics(doc):
    """Every heading of the uploaded document (no cap), or its key terms when it has no headings."""
    if not doc:
        return []
    an = docgen.analyse(doc)
    return list(an.get("headings") or an["topics"])


def options(doc=None):
    """Flat list of everything the user may choose from (document headings, chapters and subtopics)."""
    out = [{"id": f"d{i}", "title": t, "group": f"From your document ({doc['name']})"} for i, t in enumerate(doc_topics(doc))]
    for u in syllabus.tree():
        for ch in u["chapters"]:
            out.append({"id": ch["id"], "title": ch["title"], "group": u["name"]})
            out += [{"id": s["id"], "title": f"{s['number']} {s['title']}", "group": u["name"]} for s in ch["sections"]]
    return out


def _from_node(nd):
    """syllabus.node() / syllabus.locate() result -> the dict the builder works with."""
    n, ch = nd["node"], nd["chapter"]
    if nd["kind"] == "section":
        return {"id": n["id"], "title": f"{n['number']} {n['title']}", "kind": "section", "chapter": n, "query": n["title"] + " " + ch["title"]}
    return {"id": f"c{ch['id']}", "title": ch["title"], "kind": "chapter", "chapter": ch, "query": syllabus.query_for(ch)}


def resolve(ref, doc=None):
    """'c5' / 's5.2' / 'd2' / free text  ->  {id, title, kind, chapter|section node, query}  or None if it is not in the syllabus or the uploaded file."""
    ref = str(ref).strip()
    nd = syllabus.node(ref)
    if nd:
        return _from_node(nd)
    m = re.fullmatch(r"d(\d{1,3})", ref)
    if m:
        ts = doc_topics(doc)
        return {"id": ref, "title": ts[int(m.group(1))], "kind": "doc", "chapter": None, "query": ts[int(m.group(1))]} if int(m.group(1)) < len(ts) else None
    words = set(guard.content_words(ref))
    if doc and words:                                                    # a heading of the uploaded document?
        best = max(((len(words & set(guard.content_words(t))), i, t) for i, t in enumerate(doc_topics(doc))), default=(0, 0, ""))
        if best[0] >= 1 and best[0] >= len(words) / 2:
            return {"id": f"d{best[1]}", "title": best[2], "kind": "doc", "chapter": None, "query": best[2]}
    loc = syllabus.locate(ref)
    if loc and loc["cov"] >= 0.5:                                        # the words match a chapter / section TITLE of the textbook
        if loc["kind"] == "section" and len(words) < 2 and not loc["exact"]:
            loc = {**loc, "kind": "chapter", "node": loc["chapter"]}     # one generic word ('search') means the whole chapter, not one arbitrary section
        r = _from_node(loc)
        return {**r, "query": r["query"] + " " + ref}
    r = syllabus.match(ref, 1)
    if r and (r[0][1] >= 2 or (r[0][1] >= 1 and guard.check_scope(ref, doc).allowed)):
        return {**_from_node({"kind": "chapter", "node": r[0][0], "chapter": r[0][0]}), "query": syllabus.query_for(r[0][0]) + " " + ref}
    return None


def split_topics(text):
    return guard.split_topics(text)


# ----------------------------------------------------------------- typed blueprints
_ITEM = re.compile(rf"(\d{{1,2}})\s*(?:x\s*)?({LV})\b", re.I)


def parse_plan(text):
    """'Search: 2 Remember, 3 Apply; Minimax: 2 Analyze'  ->  [('Search', {'Remember': 2, 'Apply': 3}), ('Minimax', {'Analyze': 2})]  (or None)."""
    body = re.sub(r"^.*?\b(?:paper|blueprint|topics?)\b[^:;\n]{0,40}:\s*", "", text, count=1, flags=re.I | re.S) if re.search(r"\b(?:paper|blueprint)\b[^:;\n]{0,40}:", text, re.I) else text
    out = []
    for seg in [s for s in re.split(r";|\n|\|", body) if s.strip()]:
        m = re.match(r"^\s*(?:and\s+)?(?P<topic>[^:()\d][^:()]*?)\s*[:\-(]\s*(?P<rest>.+?)\)?\s*$", seg)
        if not m:
            continue
        counts = {}
        for n, lv in _ITEM.findall(m["rest"]):
            counts[LV_MAP[lv.lower()]] = counts.get(LV_MAP[lv.lower()], 0) + int(n)
        if counts:
            out.append((m["topic"].strip(" ,."), counts))
    return out or None


def parse_uniform(text, find_levels):
    """'5 questions per topic on search, minimax and CSP at Apply and Analyze level'  ->  [(topic, counts), ...]  (or None)."""
    m = re.search(r"(\d{1,2})\s+questions?\s+(?:per|for each|from each|on each|of each|each)\b\s*(?:topic)?", text, re.I)
    if not m:
        return None
    n = max(1, min(MAX_CELL * 3, int(m.group(1))))
    rest = text[m.end():]
    t = re.search(r"\b(?:on|from|covering|topics?\s*:?|about|for)\s+(.+)$", rest, re.I) or re.search(r"^[\s:]*(.+)$", rest)
    if not t:
        return None
    ttext = re.sub(rf"\b(?:at|for|of|in)\s+(?:the\s+)?(?:(?:{LV})(?:\s*(?:,|and|&|/|or)\s*)?)+\s*(?:levels?|bloom)?\b.*$", "", t.group(1), flags=re.I).strip(" .,")
    topics = split_topics(ttext)
    if not topics:
        return None
    plan = docgen.distribute(n, find_levels(text) or DEFAULT_LEVELS)
    counts = {l: plan.count(l) for l in LEVELS if l in plan}
    return [(tp, dict(counts)) for tp in topics]


# ----------------------------------------------------------------- building the paper
def build(items, doc=None, marks=None, exclude=None, length=None):
    """items = [{'id' or 'text', 'counts': {level: n}}].  -> (paper dict | None, error message | None, skipped topics)."""
    merged, skipped = {}, []
    for it in items:
        r = resolve(it.get("id") or it.get("text") or "", doc)
        if not r:
            skipped.append(str(it.get("text") or it.get("id")))
            continue
        cnt = {}
        for l, n in (it.get("counts") or {}).items():
            try:
                v = int(n)
            except (TypeError, ValueError):
                continue
            if l in LEVELS:
                cnt[l] = max(0, min(MAX_CELL, v))
        slot = merged.setdefault(r["id"], {"topic": r, "counts": {}})
        for l, n in cnt.items():
            slot["counts"][l] = min(MAX_CELL, slot["counts"].get(l, 0) + n)
    if skipped:                                                          # never drop a topic silently: the caller must confirm with the user first
        return None, f"I can't build a paper on {', '.join(skipped)}: outside the supported syllabus.", skipped
    merged = {k: v for k, v in merged.items() if sum(v["counts"].values()) > 0}
    if not merged:
        return None, "Please choose at least one topic and a number of questions for it.", skipped
    if len(merged) > MAX_TOPICS:
        return None, f"A paper can have at most {MAX_TOPICS} topics. Please remove some.", skipped
    total = sum(sum(v["counts"].values()) for v in merged.values())
    if total > MAX_TOTAL:
        return None, f"That is {total} questions. A paper can have at most {MAX_TOTAL}; please reduce the numbers.", skipped
    questions, notes, blueprint, seen = [], [], [], set(exclude or [])
    for v in merged.values():
        t, cnt = v["topic"], v["counts"]
        plan = [l for l in LEVELS for _ in range(cnt.get(l, 0))]
        if t["kind"] == "doc":
            r = docgen.generate(doc, {"plan": plan, "topic": t["title"], "qtype": "Short answer", "marks": marks, "section": t["title"], "length": length})
            got = [{"question": q["question"], "level": q["level"], "difficulty": q["difficulty"], "source": q.get("source") or t["title"],
                    "answer": q.get("answer"), "reference": q.get("reference", ""), "marks": q["marks"], "tokens": q.get("tokens")} for q in r["questions"]]
            notes += r.get("notes", [])
        else:
            got, n2 = engine.generate_for_topic(t["title"], plan, t["query"], seen, t["chapter"], length)
            for g in got:
                g["source"] = f"Section {t['chapter']['number']}" if t["kind"] == "section" else f"Chapter {t['chapter'].get('number', t['chapter']['id'])}"
                g["answer"], g["reference"] = None, ""
            notes += n2
        got.sort(key=lambda g: LEVELS.index(g["level"]))
        for g in got:
            seen.add(g["question"])
            questions.append({**g, "type": "Short answer", "options": None, "section": t["title"],
                              "marks": marks or g.get("marks") or PAPER_MARKS[g["level"]]})
        blueprint.append({"id": t["id"], "topic": t["title"], "counts": cnt, "got": len(got), "wanted": sum(cnt.values())})
    if not questions:
        return None, "I couldn't find questions for those topics. Try different topics or upload your own material.", skipped
    for i, q in enumerate(questions, 1):
        q["no"] = i
    adapted = 0
    merged_notes = []
    for n in notes:                                                        # one summary line instead of one per topic
        m = re.match(r"(\d+) question\(s\) on .+ were adapted", n)
        if m:
            adapted += int(m.group(1))
        else:
            merged_notes.append(n)
    if adapted:
        merged_notes.insert(0, f"{adapted} question(s) were adapted from related questions on the same topic to reach the level you chose.")
    notes = merged_notes
    levels = list(dict.fromkeys(q["level"] for q in questions))
    paper = {"title": "Question Paper", "filename": (doc["name"].rsplit(".", 1)[0] if doc else "AI_Syllabus"), "doc_id": doc["id"] if doc else None, "questions": questions,
             "total_marks": sum(q["marks"] for q in questions), "kind": "paper_build", "blueprint": blueprint,
             "spec": {"qtype": "Mixed", "n": len(questions), "levels": levels, "difficulty": None, "marks": marks},
             "source": "paper", "length": length, "notes": list(dict.fromkeys(notes))}
    return paper, None, skipped


def planner_block(doc=None, preselected=None):
    tree = syllabus.tree()
    return {"doc_topics": [{"id": f"d{i}", "title": t} for i, t in enumerate(doc_topics(doc))], "units": tree,
            "counts": {"chapters": sum(len(u["chapters"]) for u in tree), "sections": sum(len(c["sections"]) for u in tree for c in u["chapters"])},
            "levels": LEVELS, "preselected": preselected or [], "doc": doc["name"] if doc else None,
            "limits": {"total": MAX_TOTAL, "topics": MAX_TOPICS, "cell": MAX_CELL}, "marks": PAPER_MARKS}
