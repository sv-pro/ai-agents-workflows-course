# open.py
import os
import re
import json
import time
from typing import Dict, Any
from string import Template

import requests
from dotenv import load_dotenv

load_dotenv()

# -------------------- Variations --------------------
# old | new | fastest
VARIATION = os.getenv("VARIATION", "fastest")
SOURCE_URL = os.getenv(
    "SOURCE_URL",
    "https://maximilian-schwarzmueller.com/articles/gemma-3n-may-be-amazing/",
)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

MODELS = {
    "old": {
        "summarization": "gemma3:1b-it-qat",
        "extraction": "gemma3:4b-it-qat",
        "post_generation": "gemma3:27b-it-qat",
    },
    "new": {
        "summarization": "gemma3n:e2b",
        "extraction": "gemma3n:e4b",
        "post_generation": "gemma3n:e4b",
    },
    "fastest": {
        "summarization": "gemma3:1b-it-qat",
        "extraction": "gemma3:1b-it-qat",
        "post_generation": "gemma3:1b-it-qat",
    },
}

# активная модель = summarization (сливаем summary+tweet в один вызов)
ACTIVE_MODEL = MODELS[VARIATION]["summarization"]

# -------------------- Tunables ---------------------
NUM_CTX = int(os.getenv("NUM_CTX", "1536"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "220"))
TEMP = float(os.getenv("TEMP", "0.6"))
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")
USE_LLM_EXTRACTION = os.getenv("USE_LLM_EXTRACTION", "0") == "1"  # 0 = быстрый regex-парс
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "6000"))

session = requests.Session()  # reuse TCP for lower latency


# -------------------- Ollama helper ----------------
def ollama_generate(
    prompt: str,
    model: str = ACTIVE_MODEL,
    num_ctx: int = NUM_CTX,
    num_predict: int = NUM_PREDICT,
    temperature: float = TEMP,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    r = session.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()  # includes response + eval/prompt_eval metrics


# -------------------- Fetch HTML -------------------
def get_website_html(url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    # Robust decoding (avoid Ã¼ etc.)
    enc = r.encoding or getattr(r, "apparent_encoding", None) or "utf-8"
    try:
        return r.content.decode(enc, errors="ignore")
    except Exception:
        return r.text


# ----------- Fast (non-LLM) content extract --------
def extract_core_fast(html: str) -> str:
    # drop scripts/styles/nav/footer
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<nav[^>]*>.*?</nav>", " ", html)
    html = re.sub(r"(?is)<footer[^>]*>.*?</footer>", " ", html)

    # strip tags → text
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    # take head + small tail (often contains conclusion)
    head = text[:MAX_INPUT_CHARS]
    tail = "\n...\n" + text[-MAX_INPUT_CHARS // 6 :] if len(text) > MAX_INPUT_CHARS else ""
    return (head + tail).strip()


# --------------- Optional LLM extraction -----------
def extract_core_llm(html: str) -> str:
    prompt = f"""You extract the main article text from an HTML page (no menus/footers/scripts).
Return plain text only.

<html>
{html}
</html>
"""
    j = ollama_generate(
        prompt,
        model=MODELS[VARIATION]["extraction"],
        num_ctx=max(NUM_CTX, 2048),
        num_predict=NUM_PREDICT,
    )
    return j.get("response", "").strip()


# -------- robust JSON extraction helper ----------
def _first_json_object(text: str):
    """
    Robustly extract the first valid JSON object from arbitrary text.
    Tries json.JSONDecoder().raw_decode() starting at every '{'.
    Returns a Python dict or None.
    """
    dec = json.JSONDecoder()
    s = text.strip()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


# --------- One-call summary + X post (JSON) --------
GEN_PROMPT = r"""You produce STRICT JSON only. No prose, no code fences.
Schema:
{
  "summary": ["str", ...],    // 4-6 items, each ≤14 words, no emojis/hashtags
  "x_post": "str"             // ≤240 chars, max 1 emoji, only #AI hashtag if any
}

Rules:
- Output EXACTLY one JSON object matching the schema above.
- Keys: summary (array of strings), x_post (string). No extra keys.
- Escape quotes. Do NOT include backticks or markdown.
- If content is ambiguous, still return valid JSON matching the schema.

<content>
$CONTENT
</content>
"""


def summarize_and_make_x_post(content: str) -> dict:
    prompt = Template(GEN_PROMPT).safe_substitute(CONTENT=content[:MAX_INPUT_CHARS])
    j = ollama_generate(
        prompt,
        model=ACTIVE_MODEL,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        temperature=TEMP,
    )

    # perf metrics
    ec, ed = j.get("eval_count", 0), j.get("eval_duration", 1)
    pc, pd = j.get("prompt_eval_count", 0), j.get("prompt_eval_duration", 1)
    decode_tps = (ec / (ed / 1e9)) if ed else 0.0
    prefill_tps = (pc / (pd / 1e9)) if pd else 0.0
    print(f"[perf] decode_tps={decode_tps:.1f} tok/s, prefill_tps={prefill_tps:.1f} tok/s")

    raw = (j.get("response") or "").strip()

    # 1) extract first valid JSON object
    data = _first_json_object(raw)

    # 2) if still None, try to normalize curly quotes and retry
    if data is None:
        cleaned = (
            raw.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        data = _first_json_object(cleaned)

    # 3) last-resort fallback
    if not isinstance(data, dict):
        short = " ".join(raw.split())
        return {"summary": [short[:200]], "x_post": short[:240]}

    # ---- schema coercion ----
    summary = data.get("summary", [])
    x_post = data.get("x_post", "")

    # summary → list[str]
    if isinstance(summary, str):
        summary = [s.strip(" -*•") for s in summary.splitlines() if s.strip()]
    elif not isinstance(summary, list):
        summary = [str(summary)]

    summary = [s for s in (s.strip() for s in summary) if s]

    def cap_words(s, n=14):
        return " ".join(s.split()[:n])

    summary = [cap_words(s) for s in summary][:6]

    # ensure at least 4 bullets if content available
    if len(summary) < 4 and content:
        extras = [s for s in re.split(r"[.!?]\s+", content[:800]) if s.strip()]
        for e in extras:
            if len(summary) >= 4:
                break
            summary.append(cap_words(e))

    # x_post → str (collapse list, trim, cap)
    if isinstance(x_post, list):
        x_post = " ".join(str(x) for x in x_post)
    x_post = " ".join(str(x_post).split())[:240]
    x_post = x_post.rstrip(",;")

    return {"summary": summary, "x_post": x_post}


# ----------------------- Main ----------------------
def main():
    print(f"Using variation: {VARIATION} | model: {ACTIVE_MODEL}")

    t0 = time.time()
    print("Fetching website HTML...")
    html = get_website_html(SOURCE_URL)

    print("---------")
    print("Extracting core content...")
    t_ex0 = time.time()
    core = extract_core_llm(html) if USE_LLM_EXTRACTION else extract_core_fast(html)
    t_ex1 = time.time()
    print("Extracted core content (truncated view):")
    preview = core[:800] + ("\n...\n" if len(core) > 800 else "")
    print(preview)

    print("---------")
    print("Summarize + X post (single LLM call)...")
    t_llm0 = time.time()
    out = summarize_and_make_x_post(core)
    t_llm1 = time.time()

    print("Summary:")
    for b in out.get("summary", []):
        print(f"* {b}")

    print("\nX post:")
    print(out.get("x_post", ""))

    total = time.time() - t0
    print("---------")
    print(f"Extract time: {t_ex1 - t_ex0:.2f}s | LLM time: {t_llm1 - t_llm0:.2f}s | Total: {total:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
