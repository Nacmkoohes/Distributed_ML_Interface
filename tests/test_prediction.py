from services.prediction import predict_rating


def test_predict_rating_returns_float():
    result = predict_rating(12, 55)

    assert isinstance(result, float)