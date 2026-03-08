import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

print("Loading data...")
X_train, y_train = load_pickle("./output/train.pkl")
X_val, y_val = load_pickle("./output/val.pkl")
print(f"Data loaded: X_train shape={X_train.shape}")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-rf")
mlflow.sklearn.autolog()

print("Starting run...")
with mlflow.start_run():
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    print("Training...")
    rf.fit(X_train, y_train)
    print("Predicting...")
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)
    print(f"RMSE: {rmse:.4f}")

print("Done!")
