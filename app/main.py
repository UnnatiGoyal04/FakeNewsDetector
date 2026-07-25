from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.predictor import predict_news

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="A machine learning-powered REST API that classifies news articles as Fake or Real using TF-IDF and Logistic Regression.",
    version="1.0.0",
)


# Request body
class NewsRequest(BaseModel):
    text: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fake News Detection API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: NewsRequest):

    # Remove extra spaces
    text = request.text.strip()

    # Validate empty input
    if not text:
        raise HTTPException(
            status_code=400,
            detail="News text cannot be empty."
        )

    # Validate minimum length
    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="News text must be at least 20 characters long."
        )

    result = predict_news(text)

    return {
        "prediction": result["prediction"],
        "confidence": f'{result["confidence"]}%'
    }