from workers.worker import MLWorker


def test_worker_returns_prediction():
    worker = MLWorker()

    result = worker.predict(12, 55)

    assert isinstance(result, float)
    assert 1.0 <= result <= 5.0