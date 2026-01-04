import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

import numpy as np
import pandas as pd
import os
from io import BytesIO
from datetime import datetime

# SentinelHub
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

# OpenAI
from dotenv import load_dotenv
from openai import OpenAI
import base64
import json

# CLIP
import clip

# ------------------ ENV + OPENAI ------------------
load_dotenv()
client = OpenAI()

OPENAI_SCHEMA = {
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

def openai_classify_damage(pil_img: Image.Image) -> dict:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": (
                    "You are a damage assessment model for satellite or street-level images. "
                    "Classify building damage level. Return ONLY JSON. "
                    "damage_level: 0=intact/no visible damage, 1=minor, 2=major, 3=collapsed. "
                    "If no building is visible, set notes='no building visible' and use damage_level=0 with low confidence."
                )},
                {"type": "input_image", "image_url": data_url}
            ]
        }],
        text={"format": {"type": "json_schema", "json_schema": OPENAI_SCHEMA}}
    )

    return json.loads(resp.output_text)

# ------------------ SENTINEL HUB ------------------
def get_sh_config():
    config = SHConfig()
    config_path = os.path.join(os.getcwd(), ".sentinelhub", "config.json")
    if os.path.exists(config_path):
        import json as js
        with open(config_path, "r", encoding="utf-8") as f:
            creds = js.load(f)
            config.sh_client_id = creds["sh_client_id"]
            config.sh_client_secret = creds["sh_client_secret"]
            config.instance_id = creds.get("instance_id", config.instance_id)
    else:
        st.error("SentinelHub config.json not found in .sentinelhub/config.json")
    return config

def get_sentinel_image(lat, lon, date_from, date_to, width=512, height=512):
    config = get_sh_config()
    bbox = BBox([lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05], crs=CRS.WGS84)

    evalscript = """
    //VERSION=3
    return [B04, B03, B02];
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(date_from, date_to),
            mosaicking_order='mostRecent'
        )],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        bbox=bbox,
        size=(width, height),
        config=config
    )

    data = request.get_data()
    if not data:
        return None
    return Image.fromarray(data[0])

# ------------------ CLIP ZERO-SHOT (MAIN) ------------------
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return device, model, preprocess

@torch.no_grad()
def clip_zero_shot(pil_img: Image.Image, mode: str = "aerial") -> dict:
    """
    mode: 'aerial' или 'street'
    returns: verdict + probabilities
    """
    device, model, preprocess = load_clip()

    if mode == "aerial":
        texts = [
            "a real aerial photo of a collapsed building",
            "a real aerial photo of an intact building",
            "an illustration or cartoon (not a real photo)",
            "a map, infographic, or poster (not a real photo)"
        ]
    else:  # street
        texts = [
            "a real street-level photo of a collapsed building",
            "a real street-level photo of an intact building",
            "an illustration or cartoon (not a real photo)",
            "a meme or poster with text (not a real photo)"
        ]

    text_tokens = clip.tokenize(texts).to(device)
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)

    img_feat = model.encode_image(x)
    txt_feat = model.encode_text(text_tokens)

    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.T).squeeze(0)  # (4,)
    probs = sims.softmax(dim=-1).detach().cpu().numpy()

    p_collapsed = float(probs[0])
    p_intact = float(probs[1])
    p_not_photo = float(max(probs[2], probs[3]))

    if p_not_photo > max(p_collapsed, p_intact):
        verdict = "NOT_A_REAL_PHOTO"
    else:
        verdict = "COLLAPSED" if p_collapsed >= p_intact else "NOT_COLLAPSED"

    return {
        "verdict": verdict,
        "p_collapsed": p_collapsed,
        "p_not_collapsed": p_intact,
        "p_not_photo": p_not_photo,
        "texts": texts
    }

# ------------------ CLIP PROBE (OPTIONAL) ------------------
CLIP_CKPT = "clip_linear_probe_100.pth"

@st.cache_resource
def load_clip_and_probe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    if not os.path.exists(CLIP_CKPT):
        return device, clip_model, clip_preprocess, None

    ckpt = torch.load(CLIP_CKPT, map_location=device)
    feat_dim = clip_model.visual.output_dim
    probe = nn.Linear(feat_dim, 2).to(device)
    probe.load_state_dict(ckpt["classifier_state_dict"])
    probe.eval()

    return device, clip_model, clip_preprocess, probe

def clip_probe_predict(pil_img: Image.Image) -> dict:
    device, clip_model, clip_preprocess, probe = load_clip_and_probe()
    if probe is None:
        raise FileNotFoundError(f"{CLIP_CKPT} not found. Put it next to app.py.")

    x = clip_preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = probe(feats)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    return {
        "not_collapsed": float(probs[0]),
        "collapsed": float(probs[1]),
        "pred": int(probs[1] > 0.5)
    }

# ------------------ LOCAL MODEL (ResNet) OPTIONAL ------------------
@st.cache_resource
def load_local_model_if_exists():
    if not os.path.exists("model.pth"):
        return None
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load("model.pth", map_location="cpu"))
    m.eval()
    return m

local_model = load_local_model_if_exists()
cam_extractor = GradCAM(local_model, target_layer="layer4") if local_model else None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------ UI ------------------
st.set_page_config(layout="wide")
st.title("🌍 HealthOrbit – Earthquake Damage Detection")

st.sidebar.header("⚙️ Options")
use_clip_zeroshot = st.sidebar.checkbox("Use CLIP Zero-shot (MAIN)", value=True)
clip_mode = st.sidebar.selectbox("CLIP mode", ["aerial", "street"], index=0)

use_clip_probe = st.sidebar.checkbox("Use CLIP Probe (secondary)", value=True)
use_resnet = st.sidebar.checkbox("Use Local ResNet (optional)", value=bool(local_model))

use_openai = st.sidebar.checkbox("Use OpenAI (optional)", value=False)
run_openai = st.sidebar.button("Run OpenAI")

st.subheader("📡 Sentinel-2 Image or Manual Upload")
lat = st.number_input("Latitude", value=41.9981)
lon = st.number_input("Longitude", value=21.4254)
date_from = st.date_input("From", datetime(2023, 4, 1))
date_to = st.date_input("To", datetime(2023, 4, 10))

fetch = st.button("Fetch Sentinel-2 Image")

sentinel_img = None
if fetch:
    sentinel_img = get_sentinel_image(lat, lon, date_from.isoformat(), date_to.isoformat())
    if sentinel_img:
        st.image(sentinel_img, caption="Sentinel-2 Patch", use_column_width=True)
    else:
        st.warning("No image found (check coords/dates/credentials).")

uploaded = st.file_uploader("📤 Upload image (satellite or street)", type=["png", "jpg", "jpeg"])
input_image = sentinel_img or (Image.open(uploaded).convert("RGB") if uploaded else None)

# ------------------ INFERENCE ------------------
if input_image:
    st.image(input_image, caption="Input Image", use_column_width=True)

    # ======= CLIP ZERO-SHOT (MAIN) =======
    if use_clip_zeroshot:
        with st.spinner("Running CLIP zero-shot..."):
            try:
                zs = clip_zero_shot(input_image, mode=clip_mode)
                st.subheader("🧠 CLIP Zero-shot Result (MAIN)")
                st.json(zs)

                if zs["verdict"] == "COLLAPSED":
                    st.error(f"🔴 CLIP zero-shot: Collapsed (p={zs['p_collapsed']:.2f})")
                elif zs["verdict"] == "NOT_COLLAPSED":
                    st.success(f"✅ CLIP zero-shot: Not collapsed (p={zs['p_not_collapsed']:.2f})")
                else:
                    st.warning("⚠️ CLIP zero-shot: Not a real photo / infographic")
            except Exception as e:
                st.error(f"CLIP zero-shot error: {e}")

    # ======= CLIP PROBE (SECONDARY) =======
    if use_clip_probe:
        with st.spinner("Running CLIP probe..."):
            try:
                probe_out = clip_probe_predict(input_image)
                st.subheader("🧠 CLIP Probe Result (secondary)")
                st.json(probe_out)

                if probe_out["pred"] == 1:
                    st.warning(f"CLIP probe: 🔴 Collapsed (p={probe_out['collapsed']:.2f})")
                else:
                    st.info(f"CLIP probe: ✅ Not collapsed (p={probe_out['not_collapsed']:.2f})")
            except Exception as e:
                st.error(f"CLIP probe error: {e}")

    # ======= LOCAL RESNET (OPTIONAL) =======
    if use_resnet:
        if local_model is None:
            st.warning("Local ResNet disabled: model.pth not found.")
        else:
            input_tensor = transform(input_image).unsqueeze(0)
            input_tensor.requires_grad_()

            with torch.enable_grad():
                output = local_model(input_tensor)
                pred = torch.argmax(output, 1).item()
                cam = cam_extractor(pred, output)[0].detach()

            heatmap = cam.numpy().squeeze()
            threshold = 0.5
            active = np.argwhere(heatmap > threshold)
            st.write(f"🧠 ResNet CAM active cells: {len(active)}")

            if len(active) > 0:
                center = active.mean(axis=0)
                lat_cam = float(lat - center[0] * 0.001)
                lon_cam = float(lon + center[1] * 0.001)
                st.map(pd.DataFrame([(lat_cam, lon_cam)], columns=["lat", "lon"]))

            overlay = overlay_mask(
                to_pil_image(input_tensor.squeeze()),
                to_pil_image(cam, mode="F"),
                alpha=0.5
            )
            st.image(overlay, caption="Grad-CAM Attention (ResNet)", use_column_width=True)

            if pred == 0:
                st.success("✅ ResNet: Not collapsed / Intact")
            else:
                st.error("⚠️ ResNet: Collapsed")

    # ======= OPENAI (OPTIONAL) =======
    if use_openai and run_openai:
        with st.spinner("Calling OpenAI..."):
            try:
                out = openai_classify_damage(input_image)
                st.subheader("🧠 OpenAI Result (optional)")
                st.json(out)

                label_map = {0: "✅ Intact", 1: "🟡 Minor", 2: "🟠 Major", 3: "🔴 Collapsed"}
                lvl = out["damage_level"]
                conf = out["confidence"]
                st.info(f"OpenAI: **{label_map.get(lvl, lvl)}** (confidence {conf:.2f})")
                if out.get("notes"):
                    st.caption(f"Notes: {out['notes']}")
            except Exception as e:
                st.error(f"OpenAI error: {e}")

else:
    st.info("Upload an image or fetch a Sentinel-2 patch to run detection.")
