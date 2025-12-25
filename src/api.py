from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.preprocess import clean_text

app = FastAPI(title="Customer Support Ticket Auto-Triage API")

model = joblib.load("model/ticket_classifier.pkl")

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
