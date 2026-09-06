import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


DATA_PATH = "data/ratings.csv"
MODEL_PATH = "ml/model.joblib"


def train_model():
    # Load historical rating data.
    data = pd.read_csv(DATA_PATH)

    # User ID and movie ID are the input features.
    X = data[["user_id", "movie_id"]]

    # Rating is the value we want the model to predict.
    y = data["rating"]

    # Keep part of the data unseen during training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Create the regression model.
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    # Learn the relationship between inputs and ratings.
    model.fit(X_train, y_train)

    # Evaluate the model on unseen data.
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    print(f"Mean Squared Error: {mse:.4f}")

    # Save the trained model for later inference.
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()