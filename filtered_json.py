import json
import time
from typing import Dict, Any

from tqdm import tqdm

INPUT_JSON = "eccv_train.json"
OUTPUT_JSON = "buildings_filtered.json"
OUTPUT_STATS = "buildings_filtered_stats.json"

# Клучни зборови за "building candidates"
BUILDING_KEYWORDS = [
    "building", "house", "apartment", "church", "city_hall", "stadium",
    "mall", "office", "hotel", "hospital", "school", "residential",
    "tower", "bridge", "hall"
]

PLACE_KEYWORDS = [
    "building", "house", "apartment", "church", "city", "hall", "stadium",
    "school", "hospital", "office", "hotel", "residential"
]

def looks_like_building(key: str, item: Dict[str, Any]) -> bool:
    k = (key or "").lower()
    if any(w in k for w in BUILDING_KEYWORDS):
        return True

    places = item.get("places") or {}
    if isinstance(places, dict) and places:
        places_text = " ".join(map(str, places.keys())).lower()
        if any(w in places_text for w in PLACE_KEYWORDS):
            return True

    return False

def main():
    t0 = time.time()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    total = len(data)
    print(f"Loaded {total} items from {INPUT_JSON}")

    kept: Dict[str, Dict[str, Any]] = {}
    stats = {
        "total": total,
        "kept": 0,
        "dropped": 0,
        "reasons": {
            "no_url": 0,
            "not_building_candidate": 0,
            "kept": 0,
        },
        "url_check_enabled": False,
        "elapsed_seconds": None,
        "items_per_second": None,
    }

    # tqdm прогрес бар
    for key, item in tqdm(data.items(), total=total, desc="Filtering"):
        url = item.get("url")
        if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
            stats["reasons"]["no_url"] += 1
            continue

        if not looks_like_building(key, item):
            stats["reasons"]["not_building_candidate"] += 1
            continue

        kept[key] = item
        stats["reasons"]["kept"] += 1

    stats["kept"] = len(kept)
    stats["dropped"] = total - stats["kept"]

    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["items_per_second"] = round(total / elapsed, 2) if elapsed > 0 else None

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {OUTPUT_JSON} ({stats['kept']} items)")
    print(f"📊 Stats: {OUTPUT_STATS}")
    print(f"⏱ Time: {stats['elapsed_seconds']}s | Speed: {stats['items_per_second']} items/s")

if __name__ == "__main__":
    main()
