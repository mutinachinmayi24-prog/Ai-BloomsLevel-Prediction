"""Optional LLM layer (Gemini first, OpenRouter second). Everything works without keys."""
import os, json, re, time, requests

import guard

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

last_error = None  # shown in the UI when the AI call fails and we fall back


def _key(name):
    """Return the key, or None if it is missing / still the placeholder from .env.example."""
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not v or v.lower().startswith("your_") or "api_key_here" in v.lower():
        return None
    return v


def available():
    return bool(_key("GEMINI_API_KEY") or _key("OPENROUTER_API_KEY"))


_BAD = ("image", "tts", "audio", "live", "embedding", "vision", "robotics", "computer", "native", "learnlm", "gemma", "aqa")
_cache = {"models": None, "t": 0.0, "good": None}
_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _discover(key):
    """Ask Google which Flash models THIS key can call (cached for an hour). Avoids hard-coded names going stale."""
    if _cache["models"] is not None and time.time() - _cache["t"] < 3600:
        return _cache["models"]
    names = []
    try:
        r = requests.get(f"{_BASE}/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=15)
        if not r.ok:
            r = requests.get(f"{_BASE}/models", params={"pageSize": 200, "key": key}, timeout=15)
        if r.ok:
            for m in r.json().get("models", []):
                n = m.get("name", "").split("/")[-1]
                if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n and not any(w in n for w in _BAD):
                    names.append(n)
    except Exception as e:
        print(f"[llm] model discovery failed: {e}")

    def rank(n):  # stable before preview, full before lite, newest version first
        v = re.search(r"(\d+(?:\.\d+)?)", n)
        return (("preview" in n or "exp" in n), "lite" in n, -(float(v.group(1)) if v else 0))
    _cache["models"], _cache["t"] = sorted(set(names), key=rank), time.time()
    return _cache["models"]


def _call(model, key, body):
    """One generateContent request. Returns (status, text_or_None, error_message)."""
    url = f"{_BASE}/models/{model}:generateContent"
    for how in ("header", "query"):                       # newer and older key styles
        try:
            r = requests.post(url, json=body, timeout=45,
                              headers={"x-goog-api-key": key} if how == "header" else {}, params={} if how == "header" else {"key": key})
        except requests.RequestException as e:
            return 0, None, str(e)[:100]
        if r.ok:
            try:
                return 200, r.json()["candidates"][0]["content"]["parts"][0]["text"], ""
            except (KeyError, IndexError, ValueError):
                return 200, None, "empty response"
        if r.status_code not in (400, 401, 403):          # only an auth problem justifies trying the other key style
            break
    try:
        msg = r.json()["error"]["message"]
    except Exception:
        msg = r.text
    return r.status_code, None, msg[:140]


def _friendly(errors):
    codes = {s for _, s, _ in errors}
    parts = []
    if 503 in codes or 500 in codes or 502 in codes or 504 in codes:
        parts.append("Gemini is temporarily overloaded (high demand) - please try again in a minute")
    if 429 in codes:
        parts.append("free-tier quota reached - wait a minute or try again later")
    if codes & {401, 403} or any(s == 400 and "key" in m.lower() for _, s, m in errors):
        parts.append("the API key was rejected - check GEMINI_API_KEY in .env")
    if 0 in codes:
        parts.append("network timeout")
    if not parts and codes == {404}:
        parts.append("no usable Gemini model was found for this key")
    detail = ", ".join(f"{m}: {s}" for m, s, _ in errors[:4])
    return ("; ".join(parts) or "request failed") + f" ({detail})"


def _gemini(prompt, parts=None, json_mode=True, system=None):
    key = _key("GEMINI_API_KEY")
    cfg = {"temperature": 0.3 if json_mode else 0}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {"systemInstruction": {"parts": [{"text": system or guard.SYSTEM_RULES}]},      # fixed rules: the user can not change them
            "contents": [{"parts": parts or [{"text": prompt}]}], "generationConfig": cfg}
    tried, errors, budget = set(), [], [6]                # at most 6 HTTP calls per request

    def run(models):
        for model in models:
            if model in tried or budget[0] <= 0:
                continue
            tried.add(model)
            for attempt in range(2):
                budget[0] -= 1
                status, text, msg = _call(model, key, body)
                if text:
                    _cache["good"] = model
                    return text
                errors.append((model, status, msg))
                if status in (401, 403) or (status == 400 and "key" in msg.lower()):
                    budget[0] = 0                         # a bad key will not be fixed by trying other models
                    break
                if status in (0, 500, 502, 503, 504) and attempt == 0 and budget[0] > 0:
                    time.sleep(1.5)                       # temporary overload: one quick retry
                    continue
                break
        return None

    first = [m for m in [os.getenv("GEMINI_MODEL"), _cache["good"], "gemini-flash-latest", "gemini-flash-lite-latest"] if m]
    out = run(first) or run(_discover(key))
    if out:
        return out
    print("[llm] gemini attempts:", errors)
    raise RuntimeError(_friendly(errors))


def _openrouter(prompt, system=None):
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                      headers={"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}"},
                      json={"model": OPENROUTER_MODEL, "temperature": 0.3,
                            "messages": [{"role": "system", "content": system or guard.SYSTEM_RULES}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_json(prompt, system=None):
    """Return parsed JSON dict, or None if no key / every call failed (caller falls back)."""
    global last_error
    errs = []
    for name, fn in (("GEMINI_API_KEY", _gemini), ("OPENROUTER_API_KEY", _openrouter)):
        if not _key(name):
            continue
        try:
            txt = fn(prompt + "\n\nReply with a single valid JSON object only.", system=system)
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
            last_error = None
            return json.loads(txt)
        except Exception as e:
            errs.append(f"{name.split('_')[0].title()}: {str(e)[:300]}")
            print(f"[llm] {name} failed: {e}")
    last_error = "  ||  ".join(errs) or None
    return None


def out_of_scope(ai):
    """True when the model reported that the data is outside the allowed subject."""
    return isinstance(ai, dict) and ai.get("in_scope") is False


def vision_text(prompt, image_bytes, mime="image/png"):
    """Ask Gemini to read an image (used as the last-resort OCR). Returns text or None."""
    global last_error
    if not _key("GEMINI_API_KEY"):
        return None
    import base64
    try:
        out = _gemini(prompt, parts=[{"text": prompt}, {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}}], json_mode=False, system=guard.SYSTEM_OCR)
        last_error = None
        return out
    except Exception as e:
        last_error = f"Gemini: {str(e)[:300]}"
        print(f"[llm] vision failed: {e}")
        return None


if __name__ == "__main__":      # quick self-test:  python llm.py
    try:
        from dotenv import load_dotenv
        import pathlib
        load_dotenv(pathlib.Path(__file__).parent.parent / ".env", override=True, encoding="utf-8-sig")
    except ImportError:
        pass
    print("Gemini key found:    ", bool(_key("GEMINI_API_KEY")))
    print("OpenRouter key found:", bool(_key("OPENROUTER_API_KEY")))
    out = ask_json('Return {"ok": true, "word": "hello"}')
    print("RESULT:", out if out else "FAILED -> " + str(last_error))
