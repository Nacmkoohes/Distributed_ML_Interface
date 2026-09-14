from pydantic import BaseModel
from fastapi import FastAPI
import requests
from load_balancer.round_robin import RoundRobinLoadBalancer
from load_balancer.least_connections import LeastConnectionsLoadBalancer
from workers.worker import MLWorker
from prometheus_fastapi_instrumentator import Instrumentator
import  os

app=FastAPI()
Instrumentator().instrument(app).expose(app)

class PredictionRequest(BaseModel):
    user_id:int
    movie_id:int

workers = [
    "http://worker1:8000",
    "http://worker2:8000",
    "http://worker3:8000",
]

strategy = os.getenv("LOAD_BALANCER", "round_robin")

if strategy == "least_connections":
    load_balancer = LeastConnectionsLoadBalancer(workers)
else:
    load_balancer = RoundRobinLoadBalancer(workers)

@app.post("/predict")
def predict(request: PredictionRequest):
    for attempt in range(2):

        worker_url = load_balancer.get_next_worker()

        if strategy == "least_connections":
            load_balancer.start_request(worker_url)

        try:
            response = requests.post(
                f"{worker_url}/predict",
                json={
                    "user_id": request.user_id,
                    "movie_id": request.movie_id,
                },
                timeout=5,
            )

            worker_result = response.json()

            return {
                "user_id": request.user_id,
                "movie_id": request.movie_id,
                "predicted_rating": worker_result["predicted_rating"],
                "worker_id": worker_result["worker_id"],
            }
        except requests.RequestException:
            if attempt==1:
                raise

        finally:
            if strategy == "least_connections":
                load_balancer.finish_request(worker_url)

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