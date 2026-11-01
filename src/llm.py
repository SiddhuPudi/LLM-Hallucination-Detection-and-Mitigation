import os
import json
import time
import hashlib
import requests
from collections import deque
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
REASONING = os.getenv("LLM_REASONING", "low")      # low / medium / high / none
URL = "https://api.groq.com/openai/v1/chat/completions"

CACHE_PATH = "data/llm_cache.jsonl"
MIN_GAP = 2.2          # seconds between calls (stays under 30 requests/min)
TPM_BUDGET = 6500      # stay safely under the 8,000 tokens/min free limit
MAX_RETRIES = 6
TIMEOUT = 60

_cache = {}
_last_call = 0.0
_window = deque()      # (timestamp, tokens) for calls in the last 60 seconds

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        for line in f:
            rec = json.loads(line)
            _cache[rec["key"]] = rec


def _key(prompt, temperature, max_tokens):
    raw = f"{MODEL}|{REASONING}|{temperature}|{max_tokens}|{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _throttle(est_tokens):
    """Sleep until both the request gap and the tokens-per-minute budget allow a call."""
    while True:
        now = time.time()
        while _window and now - _window[0][0] > 60:
            _window.popleft()
        used = sum(t for _, t in _window)
        gap = MIN_GAP - (now - _last_call)
        if used + est_tokens > TPM_BUDGET and _window:
            time.sleep(max(1.0, 60 - (now - _window[0][0]) + 0.5))
        elif gap > 0:
            time.sleep(gap)
        else:
            return


def ask_llm(prompt, temperature=0.0, max_tokens=512, with_meta=False):
    """Send a prompt to Groq. Returns the answer text.
    With with_meta=True returns a dict: text, latency, model_version, tokens."""
    global _last_call
    if not API_KEY:
        raise RuntimeError("LLM_API not found. Check your .env file.")

    key = _key(prompt, temperature, max_tokens)
    if key in _cache:
        rec = _cache[key]
        return rec if with_meta else rec["text"]

    cur_max = max_tokens
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": cur_max,
    }
    if REASONING and REASONING.lower() != "none":
        body["reasoning_effort"] = REASONING
        body["include_reasoning"] = False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    est = len(prompt) // 3 + 250

    for attempt in range(MAX_RETRIES):
        body["max_tokens"] = cur_max
        _throttle(est)
        start = time.time()
        _last_call = start
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"Network error ({type(e).__name__}); retrying...")
            time.sleep(2 ** attempt)
            continue
        latency = time.time() - start

        if r.status_code == 200:
            data = r.json()
            choice = data["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            tokens = data.get("usage", {}).get("total_tokens", est)
            _window.append((time.time(), tokens))

            # Reasoning used up all the tokens -> retry with more room
            if not text and choice.get("finish_reason") == "length" and cur_max < 2048:
                cur_max *= 2
                print(f"Empty answer (thinking used all tokens); retrying with max_tokens={cur_max}")
                continue

            rec = {
                "key": key,
                "text": text,
                "latency": latency,
                "model_version": data.get("model", MODEL),
                "tokens": tokens,
            }
            if text:
                _cache[key] = rec
                with open(CACHE_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            else:
                print("Warning: empty answer returned (not cached).")
            return rec if with_meta else rec["text"]

        if r.status_code == 400 and "reasoning" in r.text.lower():
            print("Reasoning options not accepted by this model; retrying without them.")
            body.pop("reasoning_effort", None)
            body.pop("include_reasoning", None)
            continue

        if r.status_code == 429:
            if "per day" in r.text.lower():
                raise RuntimeError(
                    "Daily limit reached. Stop and rerun tomorrow; "
                    "cached answers will not be repeated.\n" + r.text[:300]
                )
            delay = float(r.headers.get("retry-after", 20))
            print(f"429 rate limit; waiting {delay:.0f}s")
            time.sleep(delay + 1)
            continue

        if r.status_code in (500, 502, 503):
            delay = 2 ** attempt * 3
            print(f"HTTP {r.status_code}; waiting {delay}s")
            time.sleep(delay)
            continue

        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")

    raise RuntimeError("Gave up after retries. Check your connection and key.")