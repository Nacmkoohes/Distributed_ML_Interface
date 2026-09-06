import  joblib
import  pandas as pd



MODEL_PATH="ml/model.joblib"
model=joblib.load(MODEL_PATH)

def predict_rating(user_id: int, movie_id: int) -> float:
    prediction = model.predict(
        pd.DataFrame(
            [[user_id, movie_id]],
            columns=["user_id", "movie_id"]
        )
    )
    return float(prediction[0])