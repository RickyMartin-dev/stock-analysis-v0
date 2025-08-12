import os, mlflow

# Configure DagsHub MLflow (tracking) + S3 (artifacts)
# Set env: MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
#          MLFLOW_TRACKING_USERNAME=<dagshub_user>
#          MLFLOW_TRACKING_PASSWORD=<dagshub_token>
# For artifacts in S3, create experiment with artifact_location=s3://<bucket>/mlflow_artifacts

def setup(experiment_name: str, artifact_location: str|None=None):
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    if artifact_location:
        try:
            exp_id = mlflow.create_experiment(experiment_name, artifact_location)
            mlflow.set_experiment(experiment_name)
        except Exception:
            mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(experiment_name)