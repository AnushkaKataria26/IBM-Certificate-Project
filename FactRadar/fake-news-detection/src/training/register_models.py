import os
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

def register_and_set_aliases():
    client = MlflowClient()
    experiment_name = "sourcetrace"
    model_name = "sourcetrace-classifier"

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
                              order_by=["start_time DESC"])

    transformer_run = next((r for r in runs if r.data.tags.get("mlflow.runName") == "transformer_v0.1"), None)
    baseline_run = next((r for r in runs if r.data.tags.get("mlflow.runName") == "baseline_v0.1"), None)

    if not transformer_run or not baseline_run:
        print("Could not find both baseline_v0.1 and transformer_v0.1 runs.")
        return

    try:
        client.create_registered_model(model_name)
        print(f"Created registered model '{model_name}'")
    except Exception:
        print(f"Registered model '{model_name}' already exists.")

    baseline_mv = mlflow.register_model(f"runs:/{baseline_run.info.run_id}/model", model_name)
    transformer_mv = mlflow.register_model(f"runs:/{transformer_run.info.run_id}/model", model_name)

    print(f"Registered Baseline as Version {baseline_mv.version}")
    print(f"Registered Transformer as Version {transformer_mv.version}")

    # Set the currently promoted model (Transformer) to Production, and the other to Staging
    client.set_registered_model_alias(model_name, "Production", transformer_mv.version)
    client.set_registered_model_alias(model_name, "Staging", baseline_mv.version)

    print("Set aliases: Transformer -> @Production, Baseline -> @Staging")

if __name__ == "__main__":
    register_and_set_aliases()
