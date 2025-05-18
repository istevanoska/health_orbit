from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins - for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Alert(BaseModel):
    lat: float
    lon: float
    severity: str
    timestamp: str

damage_reports = []

@app.post("/api/alerts")
def receive_alerts(alerts: List[Alert]):
    damage_reports.extend(alerts)
    return {"message": f"Received {len(alerts)} alerts"}

@app.get("/api/alerts")
def get_alerts():
    return damage_reports
