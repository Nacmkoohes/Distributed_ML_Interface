from services.prediction import predict_rating


class MLWorker:
    """
    Represents one ML inference worker.

    A worker receives an inference request and delegates
    the actual prediction to the prediction service.
    """
    def __init__(self,worker_id:str):
        self.worker_id=worker_id
        self.is_healthy=True

    def predict(self, user_id: int, movie_id: int) -> float:
        return predict_rating(user_id, movie_id)

    def health_check(self)->bool:
        return  self.is_healthy
    def mark_healthy(self):
        self.is_healthy=True
    def mark_unhealthy(self):
        self.is_healthy=False