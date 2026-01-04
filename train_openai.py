import json
import time
import base64
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Set, Tuple

import requests
from tqdm import tqdm
from openai import OpenAI

# ---------- CONFIG ----------
INPUT_JSON = "buildings_filtered.json"

OUT_ALL = "pred_all.jsonl"
OUT_COLLAPSED = "collapsed_only.jsonl"
OUT_ERRORS = "errors.jsonl"

MODEL = "gpt-4o-mini"
MAX_WORKERS = 16          # 12–24 пробај; ако добиваш 429 намали
CONF_THRESH = 0.60        # за collapsed_only
MAX_ITEMS: Optional[int] = None  # пр. 2000 за тест, па None

HTTP_TIMEOUT = 15
OPENAI_MAX_RETRIES = 6
# ----------------------------

client = OpenAI()
session = requests.Session()

JSON_SCHEMA = {
    "name": "damage_label",
    "schema": {
        "type": "object",
        "properties": {
            "damage_level": {"type": "integer", "minimum": 0, "maximum": 3},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "notes": {"type": "string"}
        },
        "required": ["damage_level", "confidence"],
        "additionalProperties": False
    }
}

# Краток prompt = побрзо
PROMPT = (
    "Classify building damage. "
    "damage_level: 0=intact, 1=minor, 2=major, 3=collapsed. "
    "Return ONLY JSON."
)

_write_lock = threading.Lock()

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def load_done_urls(*paths: str) -> Set[str]:
    done: Set[str] = set()
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        url = rec.get("url")
                        if isinstance(url, str):
                            done.add(url)
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
    return done

def download_image_bytes(url: str) -> bytes:
    r = session.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    r.raise_for_status()
    return r.content

def openai_call_with_retries(fn, *args, **kwargs):
    """
    Retry со exponential backoff за 429/5xx и transient грешки.
    """
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            # грубо препознавање на rate limit / server errors
            is_retryable = ("429" in msg) or ("rate limit" in msg) or ("timeout" in msg) or ("5" in msg and "http" in msg)
            if not is_retryable or attempt == OPENAI_MAX_RETRIES - 1:
                raise
            # exponential backoff + jitter
            sleep_s = (2 ** attempt) * 0.5 + random.uniform(0, 0.4)
            time.sleep(sleep_s)

def classify_by_url(url: str) -> Dict[str, Any]:
    def _call():
        return client.responses.create(
            model=MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": url}
                ]
            }],
            text={"format": {"type": "json_schema", "json_schema": JSON_SCHEMA}},
        )
    resp = openai_call_with_retries(_call)
    return json.loads(resp.output_text)

def classify_by_bytes(image_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    def _call():
        return client.responses.create(
            model=MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url}
                ]
            }],
            text={"format": {"type": "json_schema", "json_schema": JSON_SCHEMA}},
        )
    resp = openai_call_with_retries(_call)
    return json.loads(resp.output_text)

def process_one(key: str, item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("url")
    if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
        raise ValueError("Missing/invalid url")

    # 1) најбрзо: директен URL
    try:
        pred = classify_by_url(url)
        return {"key": key, "url": url, "prediction": pred, "mode": "url"}
    except Exception:
        # 2) fallback: download -> bytes
        img = download_image_bytes(url)
        pred = classify_by_bytes(img)
        return {"key": key, "url": url, "prediction": pred, "mode": "bytes_fallback"}

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    items = list(data.items())
    if MAX_ITEMS is not None:
        items = items[:MAX_ITEMS]

    done = load_done_urls(OUT_ALL, OUT_ERRORS, OUT_COLLAPSED)
    items = [(k, v) for (k, v) in items if isinstance(v, dict) and v.get("url") not in done]

    total = len(items)
    print(f"Remaining: {total} | workers={MAX_WORKERS} | model={MODEL}")

    if total == 0:
        print("Nothing to do.")
        return

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one, k, v): (k, v) for (k, v) in items}

        for fut in tqdm(as_completed(futures), total=total, desc="Classifying"):
            k, v = futures[fut]
            url = v.get("url")

            try:
                rec = fut.result()
                append_jsonl(OUT_ALL, rec)

                pred = rec.get("prediction") or {}
                level = pred.get("damage_level")
                conf = float(pred.get("confidence", 0))

                if level == 3 and conf >= CONF_THRESH:
                    append_jsonl(OUT_COLLAPSED, rec)

            except Exception as e:
                append_jsonl(OUT_ERRORS, {"key": k, "url": url, "error": str(e)})

    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.1f}s  ({total/elapsed:.2f} items/s)")
    print(f"- {OUT_ALL}\n- {OUT_COLLAPSED}\n- {OUT_ERRORS}")

if __name__ == "__main__":
    main()
