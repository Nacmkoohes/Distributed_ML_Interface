from services.prediction import predict_rating


def test_predict_rating_returns_float():
    result = predict_rating(12, 55)

    assert isinstance(result, float)

def test_prediction_is_in_valid_rating_range():
    result = predict_rating(12, 55)

    assert 1.0 <= result <= 5.0