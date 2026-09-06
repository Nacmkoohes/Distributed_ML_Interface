from services.prediction import predict_rating


class MLWorker:
    """
    Represents one ML inference worker.

    A worker receives an inference request and delegates
    the actual prediction to the prediction service.
    """

    def predict(self, user_id: int, movie_id: int) -> float:
        return predict_rating(user_id, movie_id)