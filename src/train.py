import os, io, json, math, pickle, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime
import mlflow

import warnings
warnings.filterwarnings("ignore")

from data_extractor import load_history
from features import prepare_training
from model import MLP
from mlflow_utils import setup as mlflow_setup
from utils import upload_bytes_to_s3, write_json_s3
from config import (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
S3_BUCKET, MODEL_PREFIX, MLFLOW_EXPERIMENT, tickers, ARTIFACT_LOCATION,
timeline, lookback, lr, epochs, batch_size, hidden, p, MODEL_VERSION)

import dagshub
dagshub.init(repo_owner='RickyMartin-dev', repo_name='stock-analysis-v0', mlflow=True)

BUCKET=S3_BUCKET
EXPERIMENT=MLFLOW_EXPERIMENT
TICKER=tickers

def train_once():
    # Set up MLFlow
    print('Setting Up MLFlow')
    mlflow_setup(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    # Extract Data
    print(f'Extract Data for {TICKER}')
    df = load_history(TICKER, period=timeline)
    # Prepare Training Data
    print(f'prepare training data')
    X, y, scaler = prepare_training(df)
    # Split data
    n_train = int(0.8*len(X))
    Xtr, Xva = X[:n_train], X[n_train:]
    ytr, yva = y[:n_train], y[n_train:]
    # Modeling
    in_dim = X.shape[1]
    model = MLP(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss() # loss functino

    # define Dataset
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.float32)),
                    batch_size=64, shuffle=True)

    # Start ML RUN
    with mlflow.start_run(run_name=f"{TICKER}-{datetime.utcnow().isoformat()}"):
        print('Training Starting')
        # params
        mlflow.log_param("ticker", TICKER)
        mlflow.log_param("lookback", lookback)
        mlflow.log_param("in_dim", in_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden", hidden)
        mlflow.log_param("dropout", p)

        for epoch in range(epochs):
            if epoch % 50 == 0: print(f'    update {epoch}/{epochs}')
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                logits = model(xb)
                loss = bce(logits, yb)
                loss.backward(); opt.step()
        print('Evaluating Model')
        # evaluate
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xva, dtype=torch.float32))#.squeeze(1)
            probs = torch.sigmoid(logits).numpy()
        auc = roc_auc_score(yva, probs) if len(np.unique(yva))>1 else float('nan')
        acc = accuracy_score(yva, (probs>0.5).astype(int))
        mlflow.log_metric("val_auc", float(auc))
        mlflow.log_metric("val_acc", float(acc))
        if not math.isnan(float(auc)):
            mlflow.log_metric("val_auc", float(auc), step=epoch)

        # persist artifacts to S3
        print('Persisting to S3')
        version = os.environ.get('MODEL_VERSION') or datetime.utcnow().strftime('%Y%m%d%H%M%S')
        base = f"{MODEL_PREFIX}/{version}"
        # model
        buff = io.BytesIO(); torch.save(model.state_dict(), buff); buff.seek(0)
        upload_bytes_to_s3(BUCKET, f"{base}/model.pt", buff.getvalue())
        # scaler
        upload_bytes_to_s3(BUCKET, f"{base}/scaler.pkl", pickle.dumps(scaler))
        # meta
        meta = {"ticker": TICKER, "val_auc": float(auc), "val_acc": float(acc), "created": datetime.utcnow().isoformat()}
        upload_bytes_to_s3(BUCKET, f"{base}/meta.json", json.dumps(meta).encode())
        # manifest
        write_json_s3(BUCKET, f"manifests/stock-updown/latest.json", {"current_version": version, "s3_path": base})
        mlflow.log_artifact_local = True
        mlflow.log_dict(meta, "meta.json")
        return version, meta

if __name__ == '__main__':
    print(train_once())

# import os, json, time, math
# import mlflow, mlflow.pytorch
# from mlflow import log_metric, log_param, log_artifact
# from datetime import datetime, timedelta
# import torch
# import torch.nn as nn


# import warnings
# warnings.filterwarnings("ignore")

# from config import (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, 
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, 
# tickers, timeline, lookback, epochs, batch_size,lr,hidden,p, val_friction)
# from data_extractor import load_stock_data
# from features import build_features, create_data_set_part1, create_data_set_part2
# from model import MLP#, CalibratedSigmoid
# from evaluate import evaluate_metrics

# # 1) Configure MLflow: tracking on DagsHub; artifacts to S3
# #    Set env vars in your shell / GitHub Actions secrets:
# #    
# #    AWS_ACCESS_KEY_ID=..., AWS_SECRET_ACCESS_KEY=..., AWS_DEFAULT_REGION=...
# #    MLFLOW_S3_UPLOAD_EXTRA_ARGS='{"ServerSideEncryption":"aws:kms"}'
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  
# mlflow.set_experiment(MLFLOW_EXPERIMENT)
# mlflow.autolog()

# with mlflow.start_run(run_name=f"{tickers}-train-{datetime.utcnow().isoformat()}"):

#     # Params
#     log_param("symbol", tickers)
#     log_param("lookback", lookback)
#     log_param("epochs", epochs)
#     log_param("batch_size", batch_size)
#     log_param("lr", lr)
#     log_param("hidden", hidden)
#     log_param("dropout", p)

#     # Get ticker
#     # tickers = os.getenv("TICKERS", "AAPL,MSFT,GOOG").split(",")

#     # another option for dates
#     # end = datetime.utcnow().date()
#     # start = end - timedelta(days=365*2)

#     # call data extractor to load data
#     print('attempting to call stock data loader')
#     datum = load_stock_data(tickers, timeline)
#     print('attempting to build data')
#     lookback = int(lookback)

#     # Build Initial Features
#     X, y = build_features(datum, lookback)

#     print('X shape:',X.shape)
#     print('y shape:',y.shape, 'Mean:', y.mean())

#     # Define Model to train on
#     in_features = X.shape[1]
#     model = MLP(in_features, hidden=hidden, p=p)
#     crit = nn.BCEWithLogitsLoss()
#     opt = torch.optim.Adam(model.parameters(), lr=lr)

#     # PART 1: create data set
#     Xtrain, ytrain, Xval, yval, pipeline = create_data_set_part1(X, y, val_friction)
#     # PART 2: create data set
#     train_set, val_set = create_data_set_part2(Xtrain, ytrain, Xval, yval, batch_size)

#     # Training Loop:
#     best_val = 1e9
#     best_state = None

#     for epoch in range(epochs):
#         if epoch % 10 == 0: print(f'update {epoch}/{epochs}')
#         model.train()
#         total = 0.0
#         count = 0
#         for xb, yb in train_set:
#             opt.zero_grad()
#             logits = model(xb)
#             loss = crit(logits, yb)
#             loss.backward()
#             opt.step()
#             total += loss.item() * xb.size(0)
#             count += xb.size(0)
#         train_loss = total / max(count,1)

#         val_acc, val_auc, _, _, _ = evaluate_metrics(model, val_set)

#         log_metric("train_loss", train_loss, step=epoch)
#         log_metric("val_acc", val_acc, step=epoch)
#         if not math.isnan(val_auc):
#             log_metric("val_auc", val_auc, step=epoch)

#         # Track best model by AUC (fallback to loss if AUC is nan)
#         score = -val_acc  # smaller is better placeholder; swap if using loss
#         if best_state is None or val_acc > (-score):
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

#     print('Training Finished')


#     # print('Define Model')
#     # model = Net(input_dim=X.shape[1])
#     # print("Fitting Model")
#     # model.fit(X, y, epochs=20, batch_size=64)
#     # print("Finished fititng model")

#     # calib = CalibratedSigmoid(model).fit(X, y)

#     # metrics = evaluate_metrics(calib, X, y)
#     # for k, v in metrics.items():
#     #     mlflow.log_metric(k, float(v))

#     # mlflow.log_params({"tickers": tickers, "lookback": 10, "epochs": 20})

#     # # log model to S3 via MLflow; register in Model Registry
#     # artifact_path = "model"
#     # mv = mlflow.pytorch.log_model(
#     #     pytorch_model=calib,
#     #     artifact_path=artifact_path,
#     #     registered_model_name="stock-updown-pytorch"
#     # )
#     # # Optionally set stage based on gate
#     # gate_acc = float(os.getenv("GATE_ACC", 0.52))
#     # if metrics["accuracy"] >= gate_acc:
#     #     client = mlflow.tracking.MlflowClient()
#     #     client.set_registered_model_alias("stock-updown-pytorch", "Staging", mv.model_version)

#     # mlflow.log_text(json.dumps(metrics, indent=2), "metrics.json")