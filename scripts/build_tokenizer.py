"""Train the built-in byte-level BPE tokenizer used to count tokens offline (backend/data/bpe_vocab.json).
Usage: python scripts/build_tokenizer.py [merges]      (default 3500; a small vocabulary counts MORE tokens than large ones, i.e. it is the strict side)
"""
import re, sys, json, heapq, pathlib, collections
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "backend" / "data"
PRE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
n_merges = int(sys.argv[1]) if len(sys.argv) > 1 else 3500

text = " ".join(c["text"] for c in json.load(open(DATA / "textbook_chunks.json")))
text += " " + " ".join(pd.read_excel(DATA / "Dataset.xlsx").iloc[:, 0].astype(str))
freq = collections.Counter(p.encode("utf-8").decode("latin-1") for p in PRE.findall(text))
words = {w: [list(w), c] for w, c in freq.items()}
pairs, where = collections.Counter(), collections.defaultdict(set)
for w, (s, c) in words.items():
    for a, b in zip(s, s[1:]):
        pairs[(a, b)] += c
        where[(a, b)].add(w)
heap = [(-c, p) for p, c in pairs.items()]
heapq.heapify(heap)
merges = []
while len(merges) < n_merges and heap:
    c, p = heapq.heappop(heap)
    if -c != pairs[p]:                      # stale heap entry
        if pairs[p] > 0:
            heapq.heappush(heap, (-pairs[p], p))
        continue
    if pairs[p] < 2:
        break
    merges.append(p)
    new = p[0] + p[1]
    touched = set()
    for w in list(where[p]):
        s, cnt = words[w]
        if not any(a == p[0] and b == p[1] for a, b in zip(s, s[1:])):
            continue
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] -= cnt
        i, out = 0, []
        while i < len(s):
            if i < len(s) - 1 and s[i] == p[0] and s[i + 1] == p[1]:
                out.append(new); i += 2
            else:
                out.append(s[i]); i += 1
        words[w][0] = out
        for a, b in zip(out, out[1:]):
            pairs[(a, b)] += cnt
            where[(a, b)].add(w)
            touched.add((a, b))
    for q in touched:
        if pairs[q] > 0:
            heapq.heappush(heap, (-pairs[q], q))
json.dump({"merges": merges}, open(DATA / "bpe_vocab.json", "w"))
print(f"{len(merges)} merges from {len(freq)} distinct pieces")
