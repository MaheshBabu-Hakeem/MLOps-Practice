import os
import pickle
import click
import mlflow
import mlflow.sklearn
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # Load datasets
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    logging.basicConfig(level=logging.DEBUG)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("random-forest-taxi_test")
    try:
        mlflow.sklearn.autolog(log_models=False, log_datasets=False)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        #mlflow.log_params(rf.get_params())
        #mlflow.log_metric("rmse", rmse)
        print(f"RMSE: {rmse:.4f}")


if __name__ == '__main__':
    run_train()