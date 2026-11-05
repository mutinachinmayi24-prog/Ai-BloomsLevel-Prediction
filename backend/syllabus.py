"""The syllabus = the chapters (and section titles) of the AIMA textbook, derived from the textbook index.

Used for: subject validation (is a topic inside the syllabus?), the topic picker of the question-paper builder,
and for grounding generated questions in the right textbook passages.
"""
import json, re, pathlib, collections
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = pathlib.Path(__file__).parent / "data"
GROUPS = [(1, 2, "Part I: Artificial Intelligence"), (3, 6, "Part II: Problem-solving"), (7, 12, "Part III: Knowledge, reasoning and planning"),
          (13, 17, "Part IV: Uncertain knowledge and reasoning"), (18, 21, "Part V: Learning"), (22, 25, "Part VI: Communicating, perceiving, acting"),
          (26, 27, "Part VII: Conclusions")]
JUNK = set("chapter section figure fig page example exercise exercises summary bibliographical historical notes".split())
CH_RE = re.compile(r"Chapter (\d{1,2})\.\s+([A-Z][A-Za-z,:\- ]{3,70}?)(?=\s{2,}|\s\d|\s+Section|$)")
SEC_RE = re.compile(r"Section (\d{1,2})\.(\d{1,2})\.\s+([A-Z][A-Za-z\(\)\-, ]{3,60}?)\s+\d{1,4}\b")


def _group(n):
    return next((g for a, b, g in GROUPS if a <= n <= b), "")


def _build_inferred():
    chunks = json.load(open(DATA / "textbook_chunks.json"))
    titles, owner = collections.defaultdict(collections.Counter), []
    for c in chunks:
        votes = collections.Counter(int(n) for n, _ in CH_RE.findall(c["text"]))
        for n, t in CH_RE.findall(c["text"]):
            titles[int(n)][t.strip()] += 1
        owner.append(votes.most_common(1)[0][0] if votes else (owner[-1] if owner else 1))
    texts, secs = collections.defaultdict(list), collections.defaultdict(collections.Counter)
    for c, n in zip(chunks, owner):
        texts[n].append(c["text"])
        for a, b, t in SEC_RE.findall(c["text"]):
            if int(a) == n:
                secs[n][(f"{a}.{b}", t.strip())] += 1
    nums = sorted(titles)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-\*]{2,}\b", sublinear_tf=True, max_df=0.6)
    X = vec.fit_transform([" ".join(texts[n]) for n in nums])
    names = vec.get_feature_names_out()
    out = []
    for i, n in enumerate(nums):
        row = X[i].toarray()[0]
        kws = [names[j] for j in row.argsort()[::-1][:40] if not (set(names[j].split()) & JUNK) and not re.search(r"\d", names[j])][:14]
        def clean(t):   # real section titles repeat in the page headers and are Title Case; drop sentence fragments
            ws = [w for w in t.split() if len(w) > 3]
            return len(t.split()) <= 8 and sum(w[0].isupper() for w in ws) >= max(1, 0.5 * len(ws))
        sections = [{"number": k, "title": t} for (k, t), c in sorted(secs[n].items(), key=lambda kv: [int(x) for x in kv[0][0].split(".")]) if c >= 2 and clean(t)]
        out.append({"id": n, "title": titles[n].most_common(1)[0][0], "group": _group(n), "keywords": kws, "sections": sections,
                    "chunks": [k for k, o in enumerate(owner) if o == n]})
    return out


def _build_from_toc():
    """The syllabus exactly as printed in the textbook's table of contents (data/syllabus.json, built by scripts/build_syllabus.py)."""
    toc = json.load(open(DATA / "syllabus.json"))
    chunks = json.load(open(DATA / "textbook_chunks.json"))
    span = []
    for c in chunks:                                              # page label "118" or "118-119" -> (first, last) PDF page of the passage
        nums = [int(x) for x in re.findall(r"\d+", c["page"])]
        span.append((nums[0], nums[-1]))
    inside = lambda lo, hi: [k for k, (a, b) in enumerate(span) if a <= hi and b >= lo]       # passages that overlap the page range
    chapters = []
    for u in toc["units"]:
        for c in u["chapters"]:
            sections = []
            for x in c["sections"]:
                sections.append({"id": f"s{x['number']}", "number": x["number"], "title": x["title"], "start": x["start"], "end": x["end"],
                                 "chunks": inside(x["start"], x["end"]), "chapter": c["id"]})
            chapters.append({"id": c["id"], "number": str(c["number"]), "title": c["title"], "group": u["name"], "start": c["start"], "end": c["end"],
                             "sections": sections, "chunks": inside(c["start"], c["end"])})
    for c in chapters:
        for x in c["sections"]:
            x["chunks"] = x["chunks"] or c["chunks"]              # a very short section borrows its chapter's passages
    texts = [" ".join(chunks[k]["text"] for k in c["chunks"]) for c in chapters]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-\*]{2,}\b", sublinear_tf=True, max_df=0.6)
    X = vec.fit_transform(texts)
    names = vec.get_feature_names_out()
    for i, c in enumerate(chapters):
        row = X[i].toarray()[0]
        c["keywords"] = [names[j] for j in row.argsort()[::-1][:40] if not (set(names[j].split()) & JUNK) and not re.search(r"\d", names[j])][:14]
    return chapters


try:
    CHAPTERS = _build_from_toc()
except (OSError, KeyError, ValueError):                           # no syllabus.json: fall back to what the page headers reveal
    CHAPTERS = _build_inferred()
BY_ID = {c["id"]: c for c in CHAPTERS}
SECTIONS = {x["id"]: x for c in CHAPTERS for x in c["sections"] if "id" in x}


def tree():
    """The whole syllabus, unit by unit: [{name, chapters: [{id, number, title, sections: [{id, number, title}]}]}] (every topic and subtopic)."""
    units, order = [], {}
    for c in CHAPTERS:
        u = order.get(c["group"])
        if u is None:
            u = order[c["group"]] = {"name": c["group"], "chapters": []}
            units.append(u)
        u["chapters"].append({"id": f"c{c['id']}", "number": c.get("number", str(c["id"])), "title": c["title"],
                              "sections": [{"id": x["id"], "number": x["number"], "title": x["title"]} for x in c["sections"] if "id" in x]})
    return units


def node(ref):
    """'c3' -> that chapter;  's3.5' / 'sA.1' -> that section (its chapter is under 'chapter').  None if unknown."""
    ref = str(ref).strip()
    m = re.fullmatch(r"c(\d{1,2})", ref)
    if m and int(m.group(1)) in BY_ID:
        return {"kind": "chapter", "node": BY_ID[int(m.group(1))], "chapter": BY_ID[int(m.group(1))]}
    if ref in SECTIONS:
        sec = SECTIONS[ref]
        return {"kind": "section", "node": sec, "chapter": BY_ID[sec["chapter"]]}
    return None


def locate(text):
    """Where in the textbook does this topic sit?  -> {kind, node, chapter, score, cov, exact} or None.
    Compares the topic's words with chapter and section TITLES (full coverage required), then falls back to chapter keywords."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    tw = {w for w in _words(text) if w not in SW}
    if not tw:
        return None
    best = None
    for c in CHAPTERS:
        for kind, nd in [("chapter", c)] + [("section", x) for x in c["sections"]]:
            tws = {w for w in _words(nd["title"]) if w not in SW}
            hit = tw & tws
            if not hit:
                continue
            cov, prec = len(hit) / len(tw), len(hit) / len(tws)
            score = 0.7 * cov + 0.3 * prec + (0.05 if kind == "section" else 0)
            if best is None or score > best["score"]:
                best = {"kind": kind, "node": nd, "chapter": c, "score": score, "cov": cov, "exact": cov == 1.0 and prec == 1.0}
    if best and best["cov"] >= 0.5:
        return best
    m = match(text, 1)
    return {"kind": "chapter", "node": m[0][0], "chapter": m[0][0], "score": m[0][1] / 10, "cov": 0.0, "exact": False} if m else None


def describe(loc):
    """'Chapter 18: Learning from Examples, section 18.3 Learning Decision Trees'"""
    if not loc:
        return ""
    c = loc["chapter"]
    base = f"Chapter {c.get('number', c['id'])}: {c['title']}" if str(c.get("number", "")).isdigit() else c["title"]
    return base + (f", section {loc['node']['number']} {loc['node']['title']}" if loc["kind"] == "section" else "")


def picker():
    """Compact list for the question-paper builder."""
    return [{"id": f"c{c['id']}", "title": c["title"], "group": c["group"], "number": c["id"]} for c in CHAPTERS]


def query_for(chapter):
    """Search text that describes a chapter (title + its most characteristic terms)."""
    secs = " ".join(x["title"] for x in chapter["sections"][:10])
    return chapter["title"] + " " + (secs or " ".join(chapter["keywords"][:6]))


def _stem(w):
    for suf in ("ings", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _words(text):
    return {_stem(w) for w in re.findall(r"[a-z][a-z\*\-]{2,}", text.lower())}


def match(text, top=3):
    """Which chapters does this text refer to?  -> [(chapter, score, matched_terms)], best first."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    tw = {w for w in _words(text) if w not in SW}
    if not tw:
        return []
    res = []
    for c in CHAPTERS:
        title_w = {w for w in _words(c["title"]) if w not in SW}
        sec_w = {w for s in c["sections"] for w in _words(s["title"]) if w not in SW}
        kw_w = {w for k in c["keywords"] for w in _words(k)}
        hit_t, hit_s, hit_k = tw & title_w, tw & sec_w, tw & kw_w
        score = 3.0 * len(hit_t) + 2.0 * len(hit_s - hit_t) + 1.0 * len(hit_k - hit_t - hit_s)
        if score:
            res.append((c, score, sorted(hit_t | hit_s | hit_k)))
    res.sort(key=lambda r: -r[1])
    return res[:top]


def find(ref):
    """Resolve 'c3', '3', 'search', 'Chapter 5' or a title to a chapter dict (or None)."""
    ref = str(ref).strip()
    m = re.fullmatch(r"(?:c|chapter\s*)?(\d{1,2})", ref, re.I)
    if m and int(m.group(1)) in BY_ID:
        return BY_ID[int(m.group(1))]
    for c in CHAPTERS:
        if c["title"].lower() == ref.lower():
            return c
    r = match(ref, 1)
    return r[0][0] if r and r[0][1] >= 3 else None
