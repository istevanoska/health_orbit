import json
import random
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import clip

# ---------------- CONFIG ----------------
INPUT_JSON = "buildings_filtered.json"
OUTPUT_JSON = "photo_buildings_1000.json"

TARGET_CHECK = 5000 # број на слики кои се филтрираат
SEED = 42

MAX_DOWNLOAD_WORKERS = 32    # 16–48 (ако интернет дозволува)
HTTP_TIMEOUT = 12

BATCH_SIZE = 64              # 32 на CPU, 64 на GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BANNED_SUBSTRINGS = ["shutterstock", "alamy", "dreamstime", "123rf", "istockphoto"]

random.seed(SEED)
session = requests.Session()

# CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

TEXTS = [
    "a real photo of a building",
    "a real photo of a collapsed building",
    "an illustration or cartoon drawing",
    "a meme or poster with text",
    "a satellite image",
    "a real photo with no building",
]
text_tokens = clip.tokenize(TEXTS).to(DEVICE)

def is_banned(url: str) -> bool:
    u = url.lower()
    return any(b in u for b in BANNED_SUBSTRINGS)

def download_one(key: str, url: str) -> Tuple[str, str, Optional[Image.Image], Optional[str]]:
    """Return (key, url, PIL_image or None, err_code or None)"""
    try:
        r = session.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return key, url, None, "not_image"

        try:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return key, url, img, None
        except UnidentifiedImageError:
            return key, url, None, "bad_image"
        except Exception:
            return key, url, None, "bad_image"

    except Exception:
        return key, url, None, "http_fail"

def keep_rule(probs: List[float]) -> bool:
    # индекси:
    p_building = probs[0]
    p_collapsed = probs[1]
    p_illustration = probs[2]
    p_meme = probs[3]
    p_sat = probs[4]
    p_no_building = probs[5]

    photo_building = max(p_building, p_collapsed)
    not_photo = max(p_illustration, p_meme)

    # брзо ranking правило:
    if photo_building <= not_photo:
        return False
    if photo_building <= p_sat:
        return False
    if photo_building <= p_no_building:
        return False
    return True

@torch.no_grad()
def clip_batch_probs(images: List[Image.Image]) -> List[List[float]]:
    # preprocess batch
    xb = torch.stack([preprocess(img) for img in images]).to(DEVICE)
    img_feat = clip_model.encode_image(xb)
    txt_feat = clip_model.encode_text(text_tokens)

    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    sims = img_feat @ txt_feat.T              # (B, T)
    probs = sims.softmax(dim=-1).cpu().tolist()
    return probs

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    # земи кандидати
    items = []
    for k, v in data.items():
        url = (v or {}).get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")) and not is_banned(url):
            items.append((k, url))

    random.shuffle(items)
    items = items[:TARGET_CHECK]

    stats = {"checked": 0, "kept": 0, "banned_or_invalid": 0, "http_fail": 0, "not_image": 0, "bad_image": 0, "rule_fail": 0}
    out: Dict[str, Dict[str, Any]] = {}

    # --- Step 1: паралелно симнување во results листа ---
    # ќе ги собираме успешните download-и и ќе ги процесираме во batch
    downloaded: List[Tuple[str, str, Image.Image]] = []

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as ex:
        futures = [ex.submit(download_one, k, url) for (k, url) in items]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            k, url, img, err = fut.result()
            stats["checked"] += 1
            if img is None:
                if err in stats:
                    stats[err] += 1
                else:
                    stats["http_fail"] += 1
                continue
            downloaded.append((k, url, img))

    # --- Step 2: CLIP батчирање ---
    # процесирај во batch size
    for i in tqdm(range(0, len(downloaded), BATCH_SIZE), desc="CLIP scoring"):
        batch = downloaded[i:i+BATCH_SIZE]
        imgs = [x[2] for x in batch]
        probs_list = clip_batch_probs(imgs)

        for (k, url, _img), probs in zip(batch, probs_list):
            if keep_rule(probs):
                out[k] = data[k]
                out[k]["_clip_prefilter"] = {"texts": TEXTS, "probs": probs}
                stats["kept"] += 1
            else:
                stats["rule_fail"] += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(out)} -> {OUTPUT_JSON}")
    print("Stats:", stats)
    print(f"Device={DEVICE}, download_workers={MAX_DOWNLOAD_WORKERS}, batch={BATCH_SIZE}")

if __name__ == "__main__":
    main()
