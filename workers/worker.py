import time

from fastapi import FastAPI
from pydantic import BaseModel
import  os
from services.prediction import predict_rating


app = FastAPI()


class PredictionRequest(BaseModel):
    user_id: int
    movie_id: int


class MLWorker:
    """
    Represents one ML inference worker.

    A worker receives an inference request and delegates
    the actual prediction to the prediction service.
    """

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.is_healthy = True

    def predict(self, user_id: int, movie_id: int) -> float:
        # time.sleep(2)
        return predict_rating(user_id, movie_id)

    def health_check(self) -> bool:
        return self.is_healthy

    def mark_healthy(self):
        self.is_healthy = True

    def mark_unhealthy(self):
        self.is_healthy = False

worker_id=os.getenv("WORKER_ID", "worker-1")
worker = MLWorker(worker_id)


@app.post("/predict")
def predict(request: PredictionRequest):
    predicted_rating = worker.predict(
        request.user_id,
        request.movie_id
    )

    return {
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "predicted_rating": predicted_rating,
        "worker_id": worker.worker_id,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "worker_id": worker.worker_id,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "workers.worker:app",
        host="0.0.0.0",
        port=8000,
    )