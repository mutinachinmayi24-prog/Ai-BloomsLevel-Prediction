import os, pathlib
from flask import Flask, request, jsonify, send_from_directory, send_file
try:
    from dotenv import load_dotenv; load_dotenv(pathlib.Path(__file__).parent.parent / ".env", override=True, encoding="utf-8-sig")
except ImportError:
    pass
import json
import engine, llm, chat, paper, docs, docgen, ocr, guard, syllabus, branching

app = Flask(__name__, static_folder=str(pathlib.Path(__file__).parent.parent / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024      # slides and scanned papers can be large; 20 MB is generous
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@app.errorhandler(413)
def too_big(e):
    return jsonify(error="That file is too large (maximum 20 MB)."), 413


@app.errorhandler(ValueError)
def bad(e):
    return jsonify(error=str(e)), 400


def body(*req):
    d = request.get_json(silent=True) or {}
    for k in req:
        if not str(d.get(k, "")).strip():
            raise ValueError(f"'{k}' is required")
    return d


def guarded(text, kind="question"):
    """The same guardrails as the chat: injection screen, then subject check. Raises a polite 400 error."""
    text = guard.clean(text)
    d = guard.screen(text)
    if d.allowed and kind == "topic":
        t = guard.assess_topics(text)
        if t["invalid"]:
            inv = " and ".join(i["text"][:1].upper() + i["text"][1:] for i in t["invalid"])
            ok = ", ".join(v["text"] for v in t["valid"])
            raise ValueError(f"{inv} {'is' if len(t['invalid']) == 1 else 'are'} outside the supported syllabus. " +
                             (f"Please resend with only the supported topic(s): {ok}." if ok else "Please enter a relevant AI topic."))
        valid = guard.distinct_topics(t["valid"])
        if len(valid) > 1:                                    # one topic at a time (same rule as the chat)
            raise ValueError(branching.multiple_topics_reply([v["text"] for v in valid]).replace("**", ""))
    if d.allowed and kind != "topic":
        d = guard.check_scope(text, None, kind)
    if not d.allowed:
        raise ValueError(d.message)
    return text


def read_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        raise ValueError("Please choose a file to upload.")
    return docs.ingest(f.read(), os.path.basename(f.filename))


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/classic")
def classic():
    return send_from_directory(app.static_folder, "classic.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True, llm=llm.available(), ocr=ocr.engine_name(), questions=len(engine._df), textbook_chunks=len(engine._chunks))


# ---------------- chat
@app.post("/api/chat")
def chat_api():
    d = request.get_json(silent=True) or {}
    return jsonify(chat.handle(d.get("message", ""), d.get("context")))


@app.post("/api/upload")
def upload():
    """Attach button: read the file, decide whether it is a question paper or study material, and reply like the chat does."""
    try:
        ctx = json.loads(request.form.get("context") or "{}")
    except ValueError:
        ctx = {}
    return jsonify(chat.from_upload(read_upload(), ctx))


# ---------------- classic page
@app.post("/api/predict")
def predict():
    return jsonify(engine.predict(guarded(body("question")["question"])))


@app.post("/api/evaluate-question")
def eq():
    return jsonify(engine.evaluate_question(guarded(body("question")["question"])))


@app.post("/api/upgrade")
def up():
    d = body("question")
    fn = engine.degrade if d.get("direction") == "down" else engine.upgrade
    return jsonify(fn(guarded(d["question"]), d.get("target_level")))


@app.post("/api/generate")
def gen():
    d = body("topic")
    return jsonify(engine.generate(guarded(d["topic"], "topic"), d.get("level"), d.get("difficulty"), int(d.get("n", 5))))


@app.post("/api/analyze-paper")
def analyze_paper():
    return jsonify(paper.analyze_doc(read_upload()))


@app.get("/api/syllabus")
def syllabus_api():
    return jsonify(units=syllabus.tree(), chapters=len(syllabus.CHAPTERS), sections=len(syllabus.SECTIONS))


# ---------------- downloads
@app.post("/api/export-paper")
def export_paper():
    d = request.get_json(silent=True) or {}
    if not d.get("questions") or not d.get("summary"):
        raise ValueError("Nothing to export.")
    name = os.path.splitext(os.path.basename(str(d.get("filename", "question_paper"))))[0] or "question_paper"
    return send_file(paper.to_xlsx(d), as_attachment=True, download_name=f"{name}_bloom_analysis.xlsx", mimetype=XLSX)


@app.post("/api/export-questions")
def export_questions():
    d = request.get_json(silent=True) or {}
    if not d.get("questions"):
        raise ValueError("Nothing to export.")
    name = os.path.splitext(os.path.basename(str(d.get("filename", "questions"))))[0] or "questions"
    if request.args.get("fmt") == "docx":
        return send_file(docgen.to_docx(d), as_attachment=True, download_name=f"{name}_question_paper.docx", mimetype=DOCX)
    return send_file(docgen.to_xlsx(d), as_attachment=True, download_name=f"{name}_questions.xlsx", mimetype=XLSX)


if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 8000)), debug=True)
