"""Conditional branching of the chat bot.

Every time the bot has to decide what to do next, the decision is made here as an explicit, named BRANCH instead of being buried in the dialogue code.
A branch is chosen from a CONDITION (what the user's message contains, how relevant the keywords are, how many topics were asked for, what level the
question has...).  The chosen branch decides the reply text, the buttons offered and what the bot waits for next.  Its name is sent to the browser
(`branch` in every reply), so the decision can be shown during a demo and checked in tests.

    condition                                         -> branch                      what the bot does
    ------------------------------------------------------------------------------------------------------------------------------
    request names ONE supported topic                 -> topic.single                runs the request
    request names MORE THAN ONE supported topic       -> topic.multiple              asks "which one do you want to continue on?" and waits
    some topics supported, some not                   -> topic.partial               says what is out of scope, asks to continue with the rest
    no topic supported                                -> topic.rejected              refuses and asks for an AI topic
    no topic given at all                             -> topic.missing               asks for the topic
    user answers a "which topic?" question with 1 topic -> topic.chosen              continues with that topic only
    user answers it with several / all of them        -> topic.choose_again          asks again: one topic at a time
    question analysed: keyword relevance Strong       -> analysis.strong             good question; offers similar questions on that topic
    ... Moderate and keywords agree on one topic      -> analysis.moderate           suggests sharpening the wording
    ... below Strong because keywords mix topics      -> analysis.mixed_topics       asks which topic the question is really about
    ... Weak                                          -> analysis.weak               warns that no syllabus topic matches, offers the topic list
    ... no keyword left after removing common words   -> analysis.no_keywords         asks for subject words
    question level is Remember / Create / in between  -> level.lowest / level.highest / level.middle   offers upgrade / degrade / both
"""

# every branch the bot can take, with a one-line meaning (used by tests, docs and the demo label in the page)
BRANCHES = {
    "topic.single": "One supported topic: the request runs.",
    "topic.multiple": "More than one topic in the prompt: ask which one to continue with.",
    "topic.partial": "Some topics are outside the syllabus: confirm continuing with the supported one.",
    "topic.rejected": "No topic is inside the syllabus: refuse and ask for an AI topic.",
    "topic.missing": "No topic given: ask for it.",
    "topic.chosen": "The user picked one topic from the choices: continue with it.",
    "topic.cancelled": "The user cancelled the open question: nothing is generated.",
    "topic.choose_again": "The user did not pick exactly one topic: ask again, one topic at a time.",
    "analysis.strong": "Keyword relevance is Strong: the question is focused on one syllabus topic.",
    "analysis.moderate": "Keyword relevance is Moderate: suggest sharper wording.",
    "analysis.mixed_topics": "Keywords point to more than one topic: ask which one the question is about.",
    "analysis.weak": "Keyword relevance is Weak: the question does not match a syllabus topic.",
    "analysis.no_keywords": "Only common words were found: ask for subject words.",
    "slot.levels": "Level not given: ask for the Bloom level.",
    "slot.difficulty": "Difficulty not given: ask for it.",
    "slot.n": "Number of questions not given: ask for it.",
    "slot.qtype": "Question type not given: ask for it.",
    "guard.injection": "The message tries to change the bot's rules: refuse.",
    "guard.too_long": "The message is too long: ask for a shorter one.",
    "guard.off_task": "Not an exam-question task (poem, essay, code...): refuse politely.",
    "guard.off_topic": "Another subject than the AI syllabus: refuse and name it.",
    "level.lowest": "Remember level: only an upgrade makes sense.",
    "level.highest": "Create level: only a degrade makes sense.",
    "level.middle": "Middle level: upgrade or degrade.",
}


def _chip(label, text, icon=None):
    return {"label": label, "text": text, "icon": icon}


# ----------------------------------------------------------------- branch on the NUMBER OF TOPICS in a request
def topic_branch(n_valid, n_invalid):
    """How many distinct topics did the request name?  n_valid = supported topics, n_invalid = unsupported ones."""
    if n_valid == 0 and n_invalid == 0:
        return "topic.missing"
    if n_valid == 0:
        return "topic.rejected"
    if n_valid > 1:
        return "topic.multiple"
    return "topic.partial" if n_invalid else "topic.single"


def join_names(names, word="and"):
    names = [n for n in names if n]
    return names[0] if len(names) == 1 else (", ".join(names[:-1]) + f" {word} " + names[-1] if names else "")


def multiple_topics_reply(topics, invalid=()):
    """The question asked when a prompt contains more than one topic."""
    shown = join_names([f"**{t}**" for t in topics], "and")
    left = f" ({join_names([str(i) for i in invalid])} {'is' if len(invalid) == 1 else 'are'} outside the supported syllabus, so I left {'it' if len(invalid) == 1 else 'them'} out.)" if invalid else ""
    return (f"There is more than one topic in your prompt: {shown}.{left} I work on one topic at a time, "
            "so which one do you want to continue on?")


def match_choice(msg, options):
    """Which of the offered topics did the user answer with?  -> ('one', index) | ('many', None) | ('none', None).
    Understands the topic's own words, '1' / 'first' / 'second' / 'last', and 'both' / 'all' (which is NOT accepted: one topic at a time)."""
    import re
    low = re.sub(r"[^a-z0-9\s*\-]", " ", msg.lower()).strip()
    low = re.sub(r"\s+", " ", low)
    if not low:
        return "none", None
    if re.fullmatch(r"(?:both|all|all of them|all of these|everything|every one|each|the two|all topics|both of them|both topics)(?: of them)?", low):
        return "many", None
    ords = {"first": 0, "1st": 0, "one": 0, "1": 0, "second": 1, "2nd": 1, "two": 1, "2": 1, "third": 2, "3rd": 2, "three": 2, "3": 2, "fourth": 3, "4": 3,
            "last": len(options) - 1}
    m = re.fullmatch(r"(?:the |option |number |no\.? |#)?(first|1st|one|1|second|2nd|two|2|third|3rd|three|3|fourth|4|last)(?: one| topic| option)?", low)
    if m and ords[m.group(1)] < len(options):
        return "one", ords[m.group(1)]
    norm = [re.sub(r"[^a-z0-9\s*\-]", " ", o.lower()).split() for o in options]
    for i, o in enumerate(options):                                  # the topic typed in full (or its chip)
        if low == " ".join(norm[i]):
            return "one", i
    stop = {"the", "a", "an", "of", "and", "on", "about", "only", "just", "with", "to", "in", "for", "please", "continue", "go", "let", "lets", "s", "choose", "pick",
            "select", "take", "use", "i", "want", "like", "would", "let's"}
    words = [w for w in low.split() if w not in stop]
    if not words:
        return "none", None
    hits = [i for i, ws in enumerate(norm) if any(w in ws for w in words)]
    others = [i for i, ws in enumerate(norm) if i not in hits and any(w in ws for w in words)]
    if len(hits) == 1 and not others:
        return "one", hits[0]
    if len(hits) > 1:
        return "many", None
    return "none", None


# ----------------------------------------------------------------- branch on the KEYWORD RELEVANCE of an analysed question
def score_line(rel):
    name = rel.get("metric") or "Keyword relevance"
    if rel.get("status") == "ok" and (rel.get("evidence") or {}).get("given_topic"):
        verdict = {"Strong": "related to", "Moderate": "partly related to"}.get(rel["band"], "not related to")
        return f"{name}: **{rel['score']}%** ({rel['band']}), {verdict} the topic **{rel['topic']}**."
    if rel.get("status") == "ok":
        return f"{name}: **{rel['score']}%** ({rel['band']}), mainly about **{rel['topic']}**."
    return f"{name}: **not evaluated** ({rel.get('reason', '').rstrip('.')})."


def relevance_branch(rel):
    """Condition: the bag-of-words score of the question, and whether its clauses point to more than one topic."""
    if rel.get("status") != "ok":
        return "analysis.no_keywords"
    if (rel.get("evidence") or {}).get("spread"):
        return "analysis.mixed_topics"
    return {"Strong": "analysis.strong", "Moderate": "analysis.moderate"}.get(rel["band"], "analysis.weak")


def level_branch(level):
    return "level.lowest" if level == "Remember" else "level.highest" if level == "Create" else "level.middle"


LEVEL_ASK = {
    "level.lowest": " Would you like me to **upgrade** it to a higher level?",
    "level.highest": " Would you like me to **degrade** it to an easier level?",
    "level.middle": " Would you like me to **upgrade** it (harder) or **degrade** it (easier)?",
}


def after_analysis(rel, level, question):
    """-> (branch name, extra reply text, extra buttons) for the reply to an analysed question."""
    b = relevance_branch(rel)
    topic = rel.get("topic")
    chips = []
    given = (rel.get("evidence") or {}).get("given_topic")
    if given and b in ("analysis.strong", "analysis.moderate", "analysis.weak"):      # checked against the topic the user named
        extra = {"analysis.strong": " Its keywords belong to this topic.",
                 "analysis.moderate": f" Only some of its keywords belong to {given}. Using the topic's own terms would make it sharper.",
                 "analysis.weak": f" Its keywords are not about {given}." + (f" {rel['note']}" if rel.get("note") else "")}[b]
        if b != "analysis.strong":
            chips.append(_chip(f"More questions on {given}", f"Generate 5 questions on {given}", "spark"))
        return f"{b} / {level_branch(level)}", extra + LEVEL_ASK[level_branch(level)], chips
    if b == "analysis.strong":
        extra = " The keywords agree on one syllabus topic."
        if topic:
            chips.append(_chip(f"More questions on {topic}", f"Generate 5 questions on {topic}", "spark"))
    elif b == "analysis.moderate":
        extra = " The topic is only partly clear. Using the textbook's own terms would make it sharper."
    elif b == "analysis.mixed_topics":
        extra = " It mixes more than one topic (" + ", ".join(rel["evidence"]["spread"]) + "), so it is not focused. I work best with **one topic at a time**: ask me about one of them."
        chips.append(_chip("What topics can I use?", "What topics can I ask about?", "file"))
    elif b == "analysis.weak":
        extra = " The keywords do not match a syllabus topic well. Name the topic more precisely, or choose one from the syllabus."
        chips.append(_chip("What topics can I use?", "What topics can I ask about?", "file"))
    else:
        extra = " I need some subject words (for example *minimax*, *neural network*) to match it to a topic."
        chips.append(_chip("What topics can I use?", "What topics can I ask about?", "file"))
    return f"{b} / {level_branch(level)}", extra + LEVEL_ASK[level_branch(level)], chips
