"""Topic validation: irrelevant-only and mixed requests, confirm / decline, the AI router.  Run:  python tests/test_topic_gate.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
os.chdir(os.path.join(HERE, "..", "backend"))
import json, chat, engine, docgen, docs, llm, paperbuilder
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

print("=== 1. IRRELEVANT ONLY: 'Generate 3 questions on economics.'")
reset(); r, c = say("Generate 3 questions on economics.", {})
check("rejected gracefully (no crash, no generic error)", r["intent"] == "blocked" and not r["blocks"])
check("names 'economics' as out of scope", "economics" in r["reply"] and "outside the supported syllabus" in r["reply"], r["reply"][:120])
check("asks for a relevant AI topic", "relevant AI topic" in r["reply"] and r["reply"].rstrip().endswith("Which topic would you like?"))
check("nothing generated, question bank untouched", calls == {"gen": [], "doc": [], "plan": []}, calls)
check("remembers the request (3 questions) while waiting for a topic", c.get("pending", {}).get("ask") == "topic" and c["pending"]["slots"].get("n") == 3)
r, c = say("Decision Trees", c)
check("then a relevant topic generates the 3 questions", r["intent"] == "generate" and len(r["blocks"][0]["data"]["questions"]) == 3 and calls["gen"] == ["Decision Trees"], (r["intent"], calls))

print("\n=== 2. MIXED: 'Generate 4 questions on Decision Trees and economics.'")
reset(); r, c = say("Generate 4 questions on Decision Trees and economics.", {})
check("asks the exact confirmation question", "Economics is outside the supported syllabus. Would you like to continue with Decision Trees only?" in r["reply"], r["reply"])
check("shows where Decision Trees sits in the textbook", "Learning Decision Trees" in r["reply"] and "Chapter 18" in r["reply"], r["reply"][:140])
check("NOTHING generated before the user answers (no dataset fallback either)", calls == {"gen": [], "doc": [], "plan": []} and not r["blocks"], calls)
check("offers Yes / No", [x["text"] for x in r["suggestions"]] == ["yes", "no"])
r2, c2 = say("yes", c)
check("after YES: only Decision Trees reaches the generator", calls["gen"] == ["Decision Trees"], calls)
check("...and the 4 requested questions are returned", len(r2["blocks"][0]["data"]["questions"]) == 4)
check("...with no economics anywhere", "economics" not in json.dumps(r2["blocks"][0]["data"]).lower())

print("\n=== 3. DECLINE")
reset(); r, c = say("Generate 4 questions on Decision Trees and economics.", {}); r3, c3 = say("no", c)
check("after NO: generation stops", r3["intent"] == "cancel" and not r3["blocks"] and calls == {"gen": [], "doc": [], "plan": []}, calls)
check("pending state cleared", not c3.get("pending"))
for word in ("cancel", "nope", "no thanks"):
    reset(); r, c = say("Generate 4 questions on Decision Trees and economics.", {}); r3, _ = say(word, c)
    check(f"'{word}' also declines", r3["intent"] == "cancel" and calls["gen"] == [])
reset(); r, c = say("Generate 4 questions on Decision Trees and economics.", {}); r4, c4 = say("maybe", c)
check("an unclear short reply is asked again (nothing generated)", r4["intent"] == "ask" and "yes" in r4["reply"] and c4.get("pending") and calls["gen"] == [])
r5, _ = say("Generate 2 questions on minimax", c4)
check("a brand-new request drops the old one cleanly", r5["intent"] == "generate" and calls["gen"] == ["minimax"], calls)

print("\n=== 4. other mixed shapes")
reset(); r, c = say("Generate 5 questions on search, minimax and cooking", {})
check("several valid + one invalid -> asks WHICH ONE topic (one topic at a time), says cooking was left out", r["intent"] == "choose_topic" and "more than one topic" in r["reply"] and "which one do you want to continue on" in r["reply"] and "Cooking" in r["reply"] and calls["gen"] == [], r["reply"])
r, c = say("minimax", c); check("...picking one topic continues with that topic only", r["intent"] == "generate" and calls["gen"] == ["minimax"], calls)
reset(); r, c = say("Generate 5 questions on inflation and cooking", {}); check("two invalid topics -> rejected, both named", r["intent"] == "blocked" and "inflation" in r["reply"] and "cooking" in r["reply"] and calls["gen"] == [])
reset(); r, c = say("Generate questions", {}); r, c = say("economics and search", c)
check("answering the topic question with a mixed topic also asks first", r["intent"] == "confirm_topics" and calls["gen"] == [])
r, c = say("yes", c); check("...then continues the guided conversation with 'search' only", "search" in json.dumps(c.get("pending") or {}).lower() and "economics" not in json.dumps(c).lower())
reset(); r, c = say("Generate 3 questions on Planning and Acting in the Real World", {}); check("a full syllabus title containing 'and' is ONE topic (not split)", r["intent"] == "generate" and calls["gen"] == ["Planning and Acting in the Real World"], r["reply"][:100])

print("\n=== 5. VALID requests and offline fallback are unchanged")
reset(); r, c = say("Generate 3 hard questions on minimax at Apply level", {}); d = r["blocks"][0]["data"]
check("valid prompt runs straight away", r["intent"] == "generate" and len(d["questions"]) == 3)
check("offline question-bank fallback still works for a valid topic", d["source"] == "dataset" and not llm.available())
r, c = say("Generate 5 questions on Bayesian networks", {}); check("another valid topic", r["intent"] == "generate")

print("\n=== 6. AI router cannot smuggle a topic in")
llm.available = lambda: True
llm.ask_json = lambda p, system=None: {"intent": "generate", "topic": "inflation and Decision Trees", "n": 4}
reset(); r, c = say("hmm some stuff about pruning and money", {})
check("topic extracted by the AI is gated too", r["intent"] == "confirm_topics" and calls["gen"] == [], r["intent"])
seen = []
def spy(prompt, system=None): seen.append(prompt); return {"questions": []}
llm.ask_json = spy; r2, _ = say("yes", c)
check("the model prompt contains the supported topic only", seen and "Decision Trees" in seen[0] and "inflation" not in seen[0].lower(), seen[0][:200] if seen else "no prompt")
llm.available = lambda: False

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
