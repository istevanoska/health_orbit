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


---

# HealthOrbit – Детектор за Земјотресно Оштетени Згради

Овој проект е **мрежна апликација за машинско учење** која помага во откривање на згради оштетени од земјотреси користејќи сателитски слики.  
Апликацијата комбинира **далечинско набљудување** со **длабоко учење** за брзо означување на можни зони со рушење на згради и поддршка на брзи спасувачки акции.

---

## Објаснување

### Цел
Кога ќе се случи земјотрес, критично е да се знае **кои области имаат рушење на згради**.  
Овој проект обезбедува **AI детектор** кој може да анализира:

-  **Sentinel-2 сателитски слики** (директно од SentinelHub API)  
-  **Рачно прикачени слики** (згради во `.png`, `.jpg`, `.jpeg`)  

Системот користи **ResNet18 неурална мрежа** обучена да разликува **интактни** и **рушени** згради.

### Како Работи
1. **Внес на слика** → Корисникот избира сателитска слика (SentinelHub или прикачување на датотека).  
2. **Предвидување со модел** → Мрежата дава бинарна класификација:  
   - *Зграда која не е срушена*  
   - *Срушена зграда*  
3. **GradCAM визуализација** → Heatmap ги покажува регионите кои влијаеле на одлуката на моделот.  
4. **Мапирање на оштетувања** → Можни координати на оштетување се прикажуваат на интерактивна мапа.  
5. **Известувања** → Ако се открие оштетување, апликацијата може да испрати известување.

### 1. Клонирај репозиториум
```bash
git clone https://github.com/istevanoska/health_orbit.git
cd health_orbit
### 2. Инсталирај зависности
pip install -r requirements.txt
### 3. Стартирај ја Streamlit апликацијата
py -m streamlit run app.py

