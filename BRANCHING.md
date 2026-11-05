# Conditional branching, keyword relevance and the one-topic rule

## 1. Keyword relevance (bag-of-words) - `backend/evaluator.py`
1. **Remove common terms**: stop-words, command words (explain, describe, compare...), filler (thing, stuff...).
2. **Keep the keywords** (the bag of subject words that is left; hyphenated terms such as alpha-beta and back-propagation are matched).
3. **Match them to a topic**: each textbook chapter (or section of an uploaded file) is a bag of words. A keyword scores 1.0 if it is in the topic's title or section titles, 0.85 if it is a characteristic word of the topic, 0.65 if used mainly there, 0.30 if general, 0.12 if used only in other topics, 0 if not in the syllabus.
4. **Weight**: rare words count more than words used in every chapter.
5. **Score** = 100 x weighted support of the best topic x focus x specificity. Focus drops when the keywords belong to different topics; a single very generic keyword cannot score high.

Bands: Strong >= 65, Moderate 40-64, Weak < 40. If no keyword is left, the result is "not evaluated" with a reason (no invented number).
The older faithfulness check is still computed (`evaluator.faithfulness`) and shown only as a supporting check.

## 2. One topic at a time
* Generation request with several syllabus topics -> *"There is more than one topic in your prompt: A and B. I work on one topic at a time, so which one do you want to continue on?"* Buttons offer each topic; nothing is generated until one is chosen. "both / all" is asked again.
* A pasted question whose clauses point to different topics gets the same question (`evaluator.topic_spread`); after the choice, only that topic's clause is analysed.
* Two names for the same syllabus place (e.g. breadth-first and depth-first search) count as one topic.
* The Classic page API applies the same rule. The question-paper builder is intentionally multi-topic.

## 3. Branches (`backend/branching.py`)
Each chat reply carries `branch`, shown under the message.

| Condition | Branch | Bot action |
|---|---|---|
| one supported topic | topic.single | runs |
| more than one supported topic | topic.multiple | asks which one |
| user picks one | topic.chosen | continues with it |
| user answers "both" | topic.choose_again | asks again |
| some topics unsupported | topic.partial | asks to continue with the supported one |
| no topic supported | topic.rejected | refuses, asks for an AI topic |
| no topic given | topic.missing | asks for it |
| relevance Strong / Moderate / Weak | analysis.strong / moderate / weak | different advice and buttons |
| clauses point to several topics | analysis.mixed_topics | asks to focus on one |
| no keyword left | analysis.no_keywords | asks for subject words |
| level Remember / middle / Create | level.lowest / middle / highest | offers upgrade / both / degrade |
| injection, off-task, other subject | guard.* | refuses |

## Tests
`python tests/test_branching.py` (38 checks), plus the existing three suites.
