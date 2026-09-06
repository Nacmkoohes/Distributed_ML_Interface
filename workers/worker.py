from services.prediction import predict_rating


class MLWorker:
    """
    Represents one ML inference worker.

    A worker receives an inference request and delegates
    the actual prediction to the prediction service.
    """
    def __init__(self,worker_id:str):
        self.worker_id=worker_id

    def predict(self, user_id: int, movie_id: int) -> float:
        return predict_rating(user_id, movie_id)