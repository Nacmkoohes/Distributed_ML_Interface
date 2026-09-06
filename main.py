from pydantic import BaseModel
from fastapi import FastAPI

from load_balancer.round_robin import RoundRobinLoadBalancer
from services.prediction import predict_rating
from workers.worker import MLWorker

app=FastAPI()

class PredictionRequest(BaseModel):
    user_id:int
    movie_id:int


workers=[
    MLWorker('worker_1'),
    MLWorker('worker_2'),
    MLWorker('worker_3'),
]
load_balancer=RoundRobinLoadBalancer(workers)

@app.post("/predict")
def predict(request: PredictionRequest):
    worker=load_balancer.get_next_worker()

    predicted_rating = worker.predict(
        request.user_id,
        request.movie_id
    )

    return {
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "predicted_rating": predicted_rating,
        'worker_id':worker.worker_id,
    }


@app.get('/health')
def health():
    return {'status':'ok'}