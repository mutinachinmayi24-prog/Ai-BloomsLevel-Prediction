"""Question-paper builder, complete topic coverage, Classic API.  Run:  python tests/test_topic_gate_paper.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
os.chdir(os.path.join(HERE, "..", "backend"))
import json, chat, engine, docgen, docs, llm, paperbuilder, syllabus, app
results = []
def check(name, cond, extra=""):
    results.append(bool(cond)); print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if (extra and not cond) else ""))
calls = {"gen": [], "doc": [], "plan": []}
_g, _d, _p = engine.generate, docgen.generate, engine.generate_for_topic
engine.generate = lambda topic, *a, **k: (calls["gen"].append(topic), _g(topic, *a, **k))[1]
docgen.generate = lambda doc, spec: (calls["doc"].append(spec.get("topic")), _d(doc, spec))[1]
engine.generate_for_topic = lambda topic, *a, **k: (calls["plan"].append(topic), _p(topic, *a, **k))[1]
def reset(): [v.clear() for v in calls.values()]
def say(m, ctx): r = chat.handle(m, ctx); return r, r["context"]

print("=== A. QUESTION PAPER, typed blueprint with a mixed topic list")
reset(); r, c = say("Question paper: Decision Trees: 2 Apply; economics: 1 Understand", {})
check("asks to confirm before building", r["intent"] == "confirm_topics" and "Economics is outside the supported syllabus. Would you like to continue with" in r["reply"], r["reply"][:140])
check("nothing built yet", calls["plan"] == [] and not r["blocks"])
r2, c2 = say("yes", c); d = r2["blocks"][0]["data"] if r2["blocks"] else {}
check("after YES the paper contains only the supported topic", r2["intent"] == "paper_build" and calls["plan"] and all("econom" not in t.lower() for t in calls["plan"]) and "econom" not in json.dumps(d).lower(), calls)
reset(); r, c = say("Question paper: Decision Trees: 2 Apply; economics: 1 Understand", {}); r3, _ = say("no", c)
check("after NO no paper is built", r3["intent"] == "cancel" and calls["plan"] == [])
reset(); r, c = say("Question paper: economics: 2 Apply", {})
check("only invalid topics -> rejected, nothing built", r["intent"] == "blocked" and "economics" in r["reply"] and calls["plan"] == [], r["intent"])

print("\n=== B. 'Create a question paper on economics and search' (topics named in words)")
r, c = say("Create a question paper on economics and search", {})
check("asks to confirm", r["intent"] == "confirm_topics" and "Economics is outside the supported syllabus" in r["reply"])
r2, _ = say("yes", c); pb = r2["blocks"][0]["data"]
check("then opens the builder with the supported topic preselected", r2["intent"] == "paper_builder" and pb["preselected"] == ["c3"], pb["preselected"])
r, c = say("Create a question paper on gardening", {}); check("only an invalid topic -> rejected", r["intent"] == "blocked")

print("\n=== C. planner ids are validated too (no silent skipping)")
r, c = say("/paperplan " + json.dumps({"topics": [{"id": "c99", "counts": {"Apply": 2}}]}), {})
check("an unknown id is rejected, not skipped", r["intent"] == "blocked" and not r["blocks"] and calls["plan"] == [])
r, c = say("/paperplan " + json.dumps({"topics": [{"id": "s18.3", "counts": {"Apply": 2}}, {"id": "c99", "counts": {"Apply": 1}}]}), {})
check("a valid + an unknown id -> asks first", r["intent"] == "confirm_topics")

print("\n=== D. SECTION (subtopic) selection reaches the paper pipeline")
reset(); payload = {"topics": [{"id": "s18.3", "counts": {"Understand": 1, "Apply": 2}}, {"id": "c3", "counts": {"Remember": 1}}], "marks": None}
r, c = say("/paperplan " + json.dumps(payload), {}); d = r["blocks"][0]["data"]
check("section and chapter both used", calls["plan"] == ["18.3 Learning Decision Trees", "Solving Problems by Searching"], calls["plan"])
check("paper has a heading for the subtopic", "18.3 Learning Decision Trees" in [q["section"] for q in d["questions"]])
dt = [q["question"] for q in d["questions"] if q["section"].startswith("18.3")]
check("the subtopic's questions are actually about decision trees", sum(any(w in q.lower() for w in ("decision", "tree", "entropy", "information gain", "split", "pruning")) for q in dt) >= 2, dt)
for q in dt: print("      -", q[:100])
check("source labelled with the section", {q["source"] for q in d["questions"] if q["section"].startswith("18.3")} == {"Section 18.3"})

print("\n=== E. COMPLETE topic coverage in the planner")
blk = paperbuilder.planner_block(None)
ch_ids = [c["id"] for u in blk["units"] for c in u["chapters"]]; sec_ids = [s["id"] for u in blk["units"] for c in u["chapters"] for s in c["sections"]]
toc = json.load(open("data/syllabus.json")); toc_ch = sum(len(u["chapters"]) for u in toc["units"]); toc_sec = sum(len(c["sections"]) for u in toc["units"] for c in u["chapters"])
check(f"every chapter in the textbook contents is offered ({len(ch_ids)}/{toc_ch})", len(ch_ids) == toc_ch == 29 and len(set(ch_ids)) == 29)
check(f"every subtopic in the textbook contents is offered ({len(sec_ids)}/{toc_sec})", len(sec_ids) == toc_sec == 157 and len(set(sec_ids)) == 157)
check("units are in textbook order (Parts I-VII + appendices)", [u["name"].split(":")[0] for u in blk["units"]] == ["Part I", "Part II", "Part III", "Part IV", "Part V", "Part VI", "Part VII", "Appendix A", "Appendix B"], [u["name"][:8] for u in blk["units"]])
check("chapter 27 (previously cut off) is present with its subtopics", "c27" in ch_ids and any(s.startswith("s27.") for s in sec_ids))
check("every offered id resolves to a real topic", all(paperbuilder.resolve(i) for i in ch_ids + sec_ids))
print("      units:", [(u["name"].split(":")[0], len(u["chapters"]), sum(len(c["sections"]) for c in u["chapters"])) for u in blk["units"]])

print("\n=== F. CLASSIC PAGE API (same rule, cannot ask, so it says what to resend)")
cl = app.app.test_client()
r = cl.post("/api/generate", json={"topic": "Decision Trees and economics"}); check("mixed topic -> 400 with clear instruction", r.status_code == 400 and "Economics is outside the supported syllabus" in r.json["error"] and "Decision Trees" in r.json["error"], r.json)
r = cl.post("/api/generate", json={"topic": "economics"}); check("irrelevant only -> 400, asks for an AI topic", r.status_code == 400 and "relevant AI topic" in r.json["error"], r.json)
r = cl.post("/api/generate", json={"topic": "Decision Trees", "n": 3}); check("valid topic -> 200", r.status_code == 200 and len(r.json["questions"]) == 3)
r = cl.get("/api/syllabus").json; check("syllabus API lists everything", r["chapters"] == 29 and r["sections"] == 157, r.get("chapters"))
check("engine-level defence: generate() itself refuses an outside topic", engine.generate("economics")["source"] == "blocked")

print("\n=== G. DOCUMENT flow: mixed topic is gated as well")
import docx as _docx
_dx = _docx.Document(); _dx.add_heading("Breadth-First Search", 1); _dx.add_paragraph("Breadth-first search is a strategy that expands the shallowest node first.")
_dx.add_heading("Depth-First Search", 1); _dx.add_paragraph("Depth-first search is an algorithm that expands the deepest node first. Backtracking is a variant that uses less memory.")
_dx.save("/tmp/_gate_doc.docx"); doc = docs.ingest(open("/tmp/_gate_doc.docx", "rb").read(), "notes.docx"); ctx = chat.from_upload(doc, {})["context"]
reset(); r, c = say("Generate 3 questions from my notes on depth-first search and cooking", ctx)
check("document request with an outside topic asks first", r["intent"] == "confirm_topics" and calls["doc"] == [] and calls["gen"] == [], r["intent"])
r2, c2 = say("yes", c); check("after YES the conversation continues with the supported topic only", r2["intent"] == "ask" and c2["pending"]["slots"].get("topic") == "depth-first search" and "cook" not in json.dumps(c2).lower(), c2.get("pending"))
r3, _ = say("generate now", c2); check("...and the document generator receives only that topic", calls["doc"] == ["depth-first search"], calls)
check("every heading of the document is offered in the builder", [t["title"] for t in paperbuilder.planner_block(doc)["doc_topics"]] == ["Breadth-First Search", "Depth-First Search"])

print("\n=== H. follow-ups still work after a confirmation")
reset(); r, c = say("Generate 4 questions on Decision Trees and economics.", {}); r, c = say("yes", c); r2, c2 = say("more", c)
check("'more' after confirm gives new questions on the supported topic", r2["intent"] == "generate" and calls["gen"][-1] == "Decision Trees", calls)
r3, _ = say("/similar " + r["blocks"][0]["data"]["questions"][0]["question"], c); check("'Generate similar' button still works", r3["intent"] == "generate", r3["intent"])
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
