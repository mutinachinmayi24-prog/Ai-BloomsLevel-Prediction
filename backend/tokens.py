"""Real token counting for the question-length limit.

* If `tiktoken` is installed, the standard OpenAI `cl100k_base` BPE tokenizer is used (exact counts for that tokenizer).
* Otherwise a built-in byte-level BPE tokenizer (data/bpe_vocab.json, trained by scripts/build_tokenizer.py) is used. Its vocabulary is deliberately
  small, so it splits text into more tokens than large-vocabulary tokenizers typically do: a question that fits here should also fit there.
Either way it is subword tokenization, never a word or character count.
"""
import re, json, pathlib

_PRE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
_state = {}


def _load():
    if _state:
        return _state
    try:                                                     # the standard tokenizer, when available (needs `pip install tiktoken`)
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        enc.encode("test")
        _state.update(name="cl100k_base (tiktoken)", enc=enc.encode)
        return _state
    except Exception:
        pass
    merges = json.load(open(pathlib.Path(__file__).parent / "data" / "bpe_vocab.json"))["merges"]
    _state.update(name=f"built-in BPE ({len(merges)} merges)", rank={tuple(m): i for i, m in enumerate(merges)}, cache={})
    return _state


def _bpe(piece, st):
    if piece in st["cache"]:
        return st["cache"][piece]
    syms = list(piece.encode("utf-8").decode("latin-1"))     # one symbol per byte
    while len(syms) > 1:
        ranks = [st["rank"].get((a, b), 1e9) for a, b in zip(syms, syms[1:])]
        i = min(range(len(ranks)), key=ranks.__getitem__)
        if ranks[i] == 1e9:
            break
        syms[i:i + 2] = [syms[i] + syms[i + 1]]
    st["cache"][piece] = len(syms)
    return len(syms)


def count(text):
    """Number of tokens in `text`."""
    st = _load()
    text = text or ""
    if "enc" in st:
        return len(st["enc"](text))
    return sum(_bpe(p, st) for p in _PRE.findall(text))


def name():
    return _load()["name"]
