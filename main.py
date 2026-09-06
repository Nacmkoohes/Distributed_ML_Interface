from pydantic import BaseModel
from fastapi import FastAPI
from services.prediction import predict_rating

app=FastAPI()

class PredictionRequest(BaseModel):
    user_id:int
    movie_id:int

@app.post("/predict")
def predict(request: PredictionRequest):
    predicted_rating = predict_rating(
        request.user_id,
        request.movie_id
    )

    return {
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "predicted_rating": predicted_rating
    }


@app.get('/health')
def health():
    return {'status':'ok'}