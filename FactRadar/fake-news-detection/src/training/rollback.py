import os
import json
import argparse
import sys
from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

def log_rollback(prev_version, new_version, reason="Manual rollback"):
    log_file = "models/rollback_log.jsonl"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_production_version": prev_version,
        "new_production_version": new_version,
        "reason": reason
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged rollback event to {log_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True, help="Target version number to rollback to")
    parser.add_argument("--reason", type=str, default="Manual rollback", help="Reason for rollback")
    args = parser.parse_args()

    client = MlflowClient()
    model_name = "sourcetrace-classifier"

    try:
        production_mv = client.get_model_version_by_alias(model_name, "Production")
        prev_prod_version = production_mv.version
    except Exception:
        print("No current Production model found.")
        prev_prod_version = "None"

    try:
        target_mv = client.get_model_version(model_name, str(args.version))
    except Exception as e:
        print(f"Failed to find target version {args.version}: {e}")
        sys.exit(1)

    print(f"Rolling back @Production to Version {target_mv.version}")
    client.set_registered_model_alias(model_name, "Production", target_mv.version)
    
    if prev_prod_version != "None" and prev_prod_version != target_mv.version:
        client.set_registered_model_alias(model_name, "Staging", prev_prod_version)
        print(f"Demoted Version {prev_prod_version} to @Staging")

    log_rollback(prev_prod_version, target_mv.version, args.reason)

if __name__ == "__main__":
    main()
