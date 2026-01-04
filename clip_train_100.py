import os, json, random, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import clip  # from openai/CLIP

# -------- CONFIG --------
INPUT_JSON = "photo_buildings_1000.json"   # или eccv_train.json ако ти е истата структура
N_SAMPLES = 100
SEED = 42

IMG_DIR = Path("clip_100_images")
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

# псевдо-label правила (само за демо):
POS_KEY_HINTS = ["collapsed", "collapse", "destroyed", "ruin", "ruins", "rubble", "debris", "implosion"]
POS_INC_HINTS = ["earthquake", "explosion", "building collapse"]
# ------------------------

random.seed(SEED)

def pseudo_label(key: str, item: Dict[str, Any]) -> int:
    """
    1 = collapsed (псевдо)
    0 = not collapsed
    """
    k = (key or "").lower()
    inc = " ".join((item.get("incidents") or {}).keys()).lower()

    if any(h in k for h in POS_KEY_HINTS):
        return 1
    if any(h in inc for h in POS_INC_HINTS):
        return 1
    return 0

def download_image(url: str, out_path: Path, timeout=15) -> bool:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False

def load_items(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # очекуваме dict: {key: {url, incidents, places}}
    items = []
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(v.get("url"), str) and v["url"].startswith(("http://", "https://")):
            items.append((k, v))
    return items

class ImgDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, label = self.rows[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    items = load_items(INPUT_JSON)
    random.shuffle(items)

    # земи 100 кандидати
    items = items[:N_SAMPLES]

    rows = []
    kept = 0
    for i, (key, item) in enumerate(tqdm(items, desc="Downloading")):
        url = item["url"]
        label = pseudo_label(key, item)
        # именувај безбедно
        fname = f"{i:03d}.jpg"
        out_path = IMG_DIR / fname
        ok = download_image(url, out_path)
        if ok:
            rows.append((str(out_path), label, key, url))
            kept += 1

    print(f"Downloaded {kept}/{N_SAMPLES} images (some URLs may be dead).")

    if kept < 20:
        print("Too few images downloaded. Try increasing N_SAMPLES or disable dead URLs.")
        return

    # split 80/20
    random.shuffle(rows)
    cut = int(0.8 * len(rows))
    train_rows = [(p, y) for (p, y, _, _) in rows[:cut]]
    val_rows   = [(p, y) for (p, y, _, _) in rows[cut:]]

    # load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # CLIP frozen

    # Dataset/DataLoader
    train_ds = ImgDataset(train_rows, preprocess)
    val_ds = ImgDataset(val_rows, preprocess)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Linear probe on top of CLIP image features
    feat_dim = model.visual.output_dim  # typically 512 for ViT-B/32
    clf = nn.Linear(feat_dim, 2).to(device)

    opt = optim.Adam(clf.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    def extract_feats(x):
        with torch.no_grad():
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    # training loop
    for epoch in range(1, EPOCHS + 1):
        clf.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            feats = extract_feats(xb)
            logits = clf(feats)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        # validation
        clf.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                feats = extract_feats(xb)
                logits = clf(feats)
                pred = logits.argmax(dim=1)
                v_correct += (pred == yb).sum().item()
                v_total += yb.size(0)

        val_acc = v_correct / v_total if v_total else 0.0
        avg_loss = total_loss / total

        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

    # save
    torch.save({
        "clip_model": "ViT-B/32",
        "classifier_state_dict": clf.state_dict(),
        "pseudo_labeling": True,
        "rows": rows,  # за да знаеш кои слики/URL-и ги користеше
    }, "clip_linear_probe_100.pth")
    print("Saved -> clip_linear_probe_100.pth")

if __name__ == "__main__":
    main()
