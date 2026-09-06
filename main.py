from pydantic import BaseModel
from fastapi import FastAPI

app=FastAPI()

class PredictionRequest(BaseModel):
    user_id:int
    movie_id:int

@app.post('/predict')
def predict(request:PredictionRequest):
    return {
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "predicted_rating": 4.2
    }



@app.get('/health')
def health():
    return {'status':'ok'}