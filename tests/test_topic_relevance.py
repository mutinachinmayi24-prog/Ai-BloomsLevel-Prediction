"""Topic relevance: unwanted-phrase removal, bag-of-words match against a TOPIC NAME, and validation of generated questions.
Run:  python tests/test_topic_relevance.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
os.chdir(os.path.join(HERE, "..", "backend"))
import evaluator as E, engine, chat, paperbuilder
results = []
def check(name, cond, extra=""):
    results.append(bool(cond)); print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if (extra and not cond) else ""))

print("=== 1. unwanted phrases are removed before the bag of words is made")
for q, phrase, gone in [("Explain in your own words how the minimax algorithm works.", "in your own words", {"words", "own"}),
                        ("With the help of a neat diagram, explain alpha-beta pruning.", "with the help of a neat diagram", {"neat", "diagram"}),
                        ("Write short notes on the Turing test.", "write short notes on", {"notes", "short"}),
                        ("Describe A* search with a suitable example.", "with a suitable example", {"suitable", "example"}),
                        ("Explain Bayesian networks. [CO3, L2] (5 marks)", "5 marks", {"marks", "co3"})]:
    body, found = E.unwanted_phrases(q)
    kept, _ = E.keywords(q)
    check(f"'{phrase}' is removed and none of its words is kept as a keyword", phrase in found and not (set(kept) & gone), (found, kept))
check("subject words stay: 'Learning from Examples' is not an unwanted phrase", "learning" in E.keywords("Explain learning from examples in AI.")[0])
check("boilerplate no longer lowers the score: 'with the help of a neat diagram' scores Strong",
      E.evaluate("With the help of a neat diagram, explain the working of alpha-beta pruning.")["band"] == "Strong")

print("\n=== 2. is the question related to the GIVEN topic name?")
T = lambda q, t: E.evaluate(q, topic=t)
related = [("Explain in your own words how the minimax algorithm works.", "Minimax"), ("Write short notes on the Turing test.", "Turing test"),
           ("Describe how alpha-beta pruning reduces the number of nodes evaluated.", "Minimax"), ("Explain how forward checking reduces the search space.", "Constraint satisfaction"),
           ("Calculate the information gain of an attribute for a training set.", "Decision trees"), ("Define an admissible heuristic.", "A* search"),
           ("Explain backpropagation in a multilayer network.", "Neural networks"), ("Explain how variable elimination performs exact inference.", "Bayesian networks"),
           ("Explain the Q-learning update rule in reinforcement learning.", "Reinforcement learning")]
bad = [(q, t, T(q, t)["score"]) for q, t in related if T(q, t)["band"] != "Strong"]
check(f"questions about the topic score Strong ({len(related) - len(bad)}/{len(related)})", not bad, bad)
unrelated = [("Explain in your own words how the minimax algorithm works.", "Bayesian networks"), ("Explain how forward checking reduces the search space.", "Decision trees"),
             ("Explain inflation and its effect on the economy.", "Neural networks"), ("Define a function.", "Search"),
             ("Define overfitting and underfitting in decision tree learning models.", "Reinforcement learning")]
bad = [(q, t, T(q, t)["score"]) for q, t in unrelated if T(q, t)["band"] != "Weak"]
check(f"questions about another topic score Weak ({len(unrelated) - len(bad)}/{len(unrelated)})", not bad, bad)
same_ch = T("Explain backpropagation in a multilayer network.", "Decision trees")
check("same chapter, different section -> Moderate, not Strong", same_ch["band"] == "Moderate", same_ch["score"])
r = T("Explain in your own words how the minimax algorithm works.", "Bayesian networks")
check("an unrelated question says which topic it is really about", "Adversarial Search" in r["note"], r["note"])
r = T("Write short notes on the Turing test.", "Turing test")
check("words of the topic name are never discarded ('test' in 'Turing test')", "test" in r["evidence"]["kept"] and r["evidence"]["name_found"], r["evidence"]["kept"])
check("generic name words count less: 'learning' alone does not make a question about reinforcement learning",
      T("State the purpose of ensemble learning methods.", "Reinforcement learning")["band"] == "Weak")
e = T("Explain the concept of the thing in detail.", "Minimax")
check("no keyword left -> 'not evaluated' with a reason, no invented number", e["status"] == "not_evaluated" and e["score"] is None and e["reason"])
check("without a topic the older keyword-relevance mode still works", E.evaluate("Explain the minimax algorithm.")["metric"] == "Keyword relevance")
check("engine.evaluate_question passes the topic through", engine.evaluate_question("Explain the minimax algorithm.", None, "Minimax")["relevance"]["metric"] == "Topic relevance")

print("\n=== 3. generated questions are validated with the bag of words")
for topic in ("hidden markov models", "reinforcement learning", "minimax", "decision trees"):
    g = engine.generate(topic, None, None, 5)
    weak = [q["question"] for q in g["questions"] if q.get("topic_band") == "Weak"]
    check(f"'{topic}': no generated question is Weak for the topic, and each carries its topic-match score", g["questions"] and not weak and all(q.get("topic_match") is not None for q in g["questions"]), weak or g)
g = engine.generate("hidden markov models", None, None, 5)
check("dropped question-bank matches are reported", g.get("filtered_out", 0) > 0 and "not about hidden markov models" in g.get("note", ""), g.get("note"))
pap, err, _ = paperbuilder.build([{"text": "Adversarial Search", "counts": {"Understand": 2, "Apply": 1}}])
check("the paper builder still fills the paper, every question validated against its topic", not err and len(pap["questions"]) == 3 and all(q.get("topic_band") != "Weak" for q in pap["questions"]), err or pap["questions"])

print("\n=== 4. chat")
r = chat.handle("Generate 5 questions on minimax at Understand level", {})
q1 = r["blocks"][0]["data"]["questions"][0]["question"]
ev = chat.handle("/evaluate " + q1, r["context"])
rel = ev["blocks"][0]["data"]["relevance"]
check("evaluating a generated question checks it against the topic it was generated for", rel["metric"] == "Topic relevance" and rel["topic"] == "minimax" and "the topic **minimax**" in ev["reply"], ev["reply"])
ev = chat.handle("Explain in your own words how the minimax algorithm works.", {})
check("a pasted question (no topic given) uses keyword relevance, with the phrase removed", ev["blocks"][0]["data"]["relevance"]["evidence"]["phrases"] == ["in your own words"])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
