from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from app.schemas import PredictionInput, PredictionOutput, Sentiment
from app.model_loader import predict_sentiment
from app.db import get_session

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, session: Session = Depends(get_session)):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Prediction input is required.")
    try:
        result = predict_sentiment(payload.text)
        entry = Sentiment(text=payload.text, prediction=result)
        session.add(entry)
        session.commit()
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )