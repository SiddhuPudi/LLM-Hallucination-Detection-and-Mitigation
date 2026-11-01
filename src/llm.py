import os
import json
import time
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API")
MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

CACHE_PATH = "data/llm_cache.jsonl"
MIN_GAP = 4.5          # seconds between real API calls (~13 per minute)
MAX_RETRIES = 6

_cache = {}
_last_call = 0.0

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        for line in f:
            rec = json.loads(line)
            _cache[rec["key"]] = rec


def _key(prompt, temperature, max_tokens):
    raw = f"{MODEL}|{temperature}|{max_tokens}|{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def ask_llm(prompt, temperature=0.0, max_tokens=1024, with_meta=False):
    """Send a prompt to Gemini. Returns the answer text.
    With with_meta=True returns a dict: text, latency, model_version."""
    global _last_call
    if not API_KEY:
        raise RuntimeError("LLM_API not found. Check your .env file.")

    key = _key(prompt, temperature, max_tokens)
    if key in _cache:
        rec = _cache[key]
        return rec if with_meta else rec["text"]

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}

    for attempt in range(MAX_RETRIES):
        # stay under the per-minute limit
        wait = MIN_GAP - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)

        start = time.time()
        _last_call = start
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=90)
        except requests.RequestException as e:
            print(f"Network error ({e}); retrying...")
            time.sleep(2 ** attempt)
            continue
        latency = time.time() - start

        if r.status_code == 200:
            data = r.json()
            text = ""
            cands = data.get("candidates", [])
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
            rec = {
                "key": key,
                "text": text,
                "latency": latency,
                "model_version": data.get("modelVersion", MODEL),
            }
            _cache[key] = rec
            with open(CACHE_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec if with_meta else text

        if r.status_code in (429, 500, 503):
            delay = min(2 ** attempt * 5, 60)
            print(f"HTTP {r.status_code}; waiting {delay}s (attempt {attempt+1})")
            time.sleep(delay)
            continue

        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")

    raise RuntimeError(
        "Gave up after retries. If you keep seeing 429, you may have hit the "
        "daily free quota. Wait for the reset (midnight Pacific) and rerun; "
        "cached answers will not be repeated."
    )