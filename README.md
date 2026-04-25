# 🌱 GreenPredict AI — React Frontend

Pixel-faithful React frontend for the GreenPredict-AI plant survival prediction system.

## 📁 Project Structure

```
src/
├── api/predict.js           ← Backend integration + local simulation fallback
├── components/
│   ├── Navbar.jsx/.css
│   ├── TreeCard.jsx/.css
│   └── TreeDetailModal.jsx/.css
├── data/plants.js           ← Plant data + Maharashtra climate data
├── pages/
│   ├── HomePage.jsx/.css
│   ├── PredictPage.jsx/.css
│   ├── ResultsPage.jsx/.css
│   └── EncyclopediaPage.jsx/.css
├── styles/globals.css       ← CSS variables, animations, resets
└── App.js                   ← Routing via useState
```

## 🚀 Setup

```bash
npm install
npm start          # http://localhost:3000
npm run build      # production build
```

## 🔗 Backend Integration

The app uses a **local simulation** by default (no backend needed).

To connect your Python backend, add a FastAPI endpoint:

```python
# api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, joblib

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = joblib.load("model/random_forest.pkl")

@app.post("/predict")
def predict(payload: dict):
    input_data = pd.DataFrame([payload])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    probs = model.predict_proba(input_data)[0]
    prob_dict = {p: float(v*100) for p, v in zip(model.classes_, probs)}
    ranked = sorted([{"plant":k,"probability":v,"rank":i+1}
                     for i,(k,v) in enumerate(sorted(prob_dict.items(),key=lambda x:-x[1]))],
                    key=lambda x:x["rank"])
    return {"predictions": prob_dict, "ranked": ranked}
```

```bash
pip install fastapi uvicorn
uvicorn api_server:app --port 8501
REACT_APP_API_URL=http://localhost:8501 npm start
```

## 🎨 Design Tokens (globals.css)

- Primary: `#3d8b47` | Dark: `#2d6a35` | BG: `#f4f8f0`
- Font: DM Sans + DM Serif Display
- Radius: 8 / 12 / 20 / 28px

## 📄 Pages

| State | Page |
|-------|------|
| `home` | Landing — hero + environmental factors |
| `predict` | Soil form with auto climate data |
| `results` | Ranked tree cards + selected tree stats |
| `encyclopedia` | Browse + search all 8 trees |
