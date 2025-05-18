import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
from io import BytesIO

# Sentinel Hub imports
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

def get_sh_config():
    config = SHConfig()
    config_path = os.path.join(os.getcwd(), ".sentinelhub", "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            creds = json.load(f)
            config.sh_client_id = creds["sh_client_id"]
            config.sh_client_secret = creds["sh_client_secret"]
            config.instance_id = creds["instance_id"]
    else:
        st.error("SentinelHub config.json not found!")
    return config


def get_sentinel_image(lat, lon, date_from, date_to, width=512, height=512):
    config = get_sh_config()
    bbox = BBox(bbox=[lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05], crs=CRS.WGS84)
    size = (width, height)

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
        size=size,
        config=config
    )

    response = request.get_data()
    if not response:
        return None
    image_data = response[0]
    return Image.fromarray(image_data)


# def send_alert_email(team_email, lat, lon, image_path):
#     msg = EmailMessage()
#     msg['Subject'] = '🚨 ALERT: Damaged Area Detected'
#     msg['From'] = 'yourapp@gmail.com'
#     msg['To'] = team_email
#     body = f"New damage detected at:\nLatitude: {lat}\nLongitude: {lon}\n\nPlease respond urgently."
#     msg.set_content(body)
#
#     try:
#         with open(image_path, 'rb') as img:
#             msg.add_attachment(img.read(), maintype='image', subtype='png', filename='damage.png')
#
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#             smtp.login('yourapp@gmail.com', 'your_app_password')
#             smtp.send_message(msg)
#     except FileNotFoundError:
#         st.error(f"❌ Email not sent. File {image_path} not found.")


# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

cam_extractor = GradCAM(model, target_layer='layer4')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🌍 HealthOrbit – Earthquake Damage Detector")

st.subheader("🛰️ Retrieve Sentinel-2 Image or Upload Manually")
lat = st.number_input("Latitude", value=41.9981)
lon = st.number_input("Longitude", value=21.4254)
date_from = st.date_input("From Date", datetime(2023, 4, 1))
date_to = st.date_input("To Date", datetime(2023, 4, 10))
fetch_button = st.button("📥 Fetch Sentinel-2 Image")

sentinel_img = None
if fetch_button:
    sentinel_img = get_sentinel_image(lat, lon, date_from.isoformat(), date_to.isoformat())
    if sentinel_img:
        st.image(sentinel_img, caption="Sentinel-2 Patch", use_column_width=True)
    else:
        st.warning("No Sentinel-2 image found for this location and date.")

uploaded = st.file_uploader("📤 Or upload a satellite building patch", type=['png', 'jpg', 'jpeg'])
input_image = sentinel_img or (Image.open(uploaded).convert("RGB") if uploaded else None)

if input_image:
    st.image(input_image, caption="📷 Input Image", use_column_width=True)

    input_tensor = transform(input_image).unsqueeze(0)
    input_tensor.requires_grad_()

    with torch.enable_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
        activation_map = cam_extractor(pred, output)
        heatmap = activation_map[0].detach().numpy()
        heatmap = np.squeeze(heatmap)

        threshold = 0.5
        damaged_pixels = np.argwhere(heatmap > threshold)
        st.write(f"🧠 Detected {len(damaged_pixels)} high-activation pixels (possible damage).")

        deg_per_pixel = 0.00001
        damaged_coords = [
            (
                lat - row * deg_per_pixel,
                lon + col * deg_per_pixel
            )
            for row, col in damaged_pixels
        ]
        df_coords = pd.DataFrame(damaged_coords, columns=['lat', 'lon'])
        st.map(df_coords)

        result = overlay_mask(
            to_pil_image(input_tensor.squeeze()),
            to_pil_image(activation_map[0], mode='F'),
            alpha=0.5
        )

        output_path = os.path.join(os.getcwd(), "output.png")
        result.save(output_path)
        st.image(result, caption="🧠 Model Attention Map", use_column_width=True)

        if damaged_coords:
            for coord in damaged_coords[:3]:
                lat_alert, lon_alert = coord
                # send_alert_email(
                #     team_email="rescue_team@ngo.org",
                #     lat=lat_alert,
                #     lon=lon_alert,
                #     image_path=output_path
                # )

                alert_data = {
                    "lat": lat_alert,
                    "lon": lon_alert,
                    "severity": "high",
                    "timestamp": datetime.utcnow().isoformat()
                }
                try:
                    response = requests.post("http://localhost:8000/api/alerts", json=alert_data)
                    if response.status_code == 200:
                        st.success(f"✅ Alert sent to backend for {lat_alert:.6f}, {lon_alert:.6f}")
                    else:
                        st.warning(f"⚠️ Backend alert failed for {lat_alert:.6f}, {lon_alert:.6f}")
                except Exception as e:
                    st.error(f"❌ Error sending alert: {e}")

    if pred == 0:
        st.success("✅ This building appears intact.")
    else:
        st.error("⚠️ This building appears collapsed.")
