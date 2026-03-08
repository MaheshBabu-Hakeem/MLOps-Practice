import mlflow
import mlflow.sklearn

print("Setting tracking URI...")
mlflow.set_tracking_uri("http://localhost:5000")

print("Setting experiment...")
mlflow.set_experiment("test")

print("Calling autolog...")
mlflow.sklearn.autolog()

print("Autolog successful!")
