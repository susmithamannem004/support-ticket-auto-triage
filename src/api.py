from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from src.preprocess import clean_text

app = FastAPI(title="Customer Support Ticket Auto-Triage API")

# ✅ Robust model path (works with uvicorn, GitHub, deployment)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ticket_classifier.pkl")

model = joblib.load(MODEL_PATH)

class Ticket(BaseModel):
    subject: str
    description: str

@app.post("/predict")
def predict(ticket: Ticket):
    text = clean_text(ticket.subject + " " + ticket.description)
    prediction = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])

    return {
        "predicted_category": prediction,
        "confidence": round(confidence, 2)
    }
