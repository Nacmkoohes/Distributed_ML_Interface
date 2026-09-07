from pydantic import BaseModel
from fastapi import FastAPI
import requests
from load_balancer.round_robin import RoundRobinLoadBalancer
from workers.worker import MLWorker

app=FastAPI()

class PredictionRequest(BaseModel):
    user_id:int
    movie_id:int

workers = [
    "http://worker1:8000",
    "http://worker2:8000",
    "http://worker3:8000",
]

load_balancer = RoundRobinLoadBalancer(workers)



@app.post("/predict")
def predict(request: PredictionRequest):
    worker_url=load_balancer.get_next_worker()
    response = requests.post(
        f"{worker_url}/predict",
        json={
            "user_id": request.user_id,
            "movie_id": request.movie_id,
        },
    )

    worker_result = response.json()
    return {
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "predicted_rating": worker_result['predicted_rating'],
        'worker_id':worker_result['worker_id'],
    }


@app.get('/health')
def health():
    return {'status':'ok'}

if __name__=='__main__':
    import  uvicorn

    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000

    )