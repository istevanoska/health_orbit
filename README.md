# HealthOrbit – Earthquake Damage Detector

This project is a **machine learning web application** designed to assist in detecting earthquake-damaged buildings using satellite imagery.  
The app combines **remote sensing** with **deep learning** to quickly highlight areas of possible building collapse, supporting faster response and rescue efforts.  

---

## Explanation

### Goal
When an earthquake strikes, it is critical to know **which areas suffered building collapses**. This project provides an **AI-powered detector** that can analyze either:
- **Sentinel-2 satellite images** (fetched directly from the SentinelHub API), or  
- **Manually uploaded images** (building patches in `.png`, `.jpg`, `.jpeg`)  

The system then uses a **ResNet18 neural network** trained to distinguish between **intact** and **collapsed** buildings.

### How It Works
1. **Input Image** → The user selects a satellite patch (via SentinelHub or file upload).  
2. **Model Prediction** → The neural network outputs a binary classification:  
   - *Intact building*  
   - *Collapsed building*  
3. **GradCAM Visualization** → A heatmap highlights **which regions** influenced the model’s decision.  
4. **Damage Mapping** → Potentially damaged coordinates are plotted on an interactive map.  
5. **Alerts** → If damage is detected, the app can send alerts (with location and severity) to a backend rescue system.  

This workflow simulates a **decision-support tool for disaster management teams**.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/istevanoska/health_orbit.git
cd /health_orbit
### 2. Install the dependencies
pip install -r requirements.txt
### 3. Start the Streamlit app
py -m streamlit run app.py
If u have a Linux or a Mac
streamlit run app.py
### After running, open your browser at
http://localhost:8501
