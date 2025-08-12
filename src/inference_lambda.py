import os, json, io, pickle, math
import numpy as np, torch
from data_extractor import load_history
from features import prepare_inference
from model import MLP
from utils import read_json_s3, download_s3_to_bytes, Timer, logger, put_metric

from dotenv import load_dotenv

load_dotenv()

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.environ['AWS_S3_BUCKET']
PREFIX = os.environ.get('MODEL_PREFIX','models/stock-updown')
MODEL_VERSION = os.environ.get('MODEL_VERSION')  # pinned by published Lambda version
NAMESPACE = os.environ.get('METRIC_NAMESPACE','StockUpDown')

_model, _scaler, _in_dim, _resolved_version = None, None, None, None

def _load_artifacts():
    global _model, _scaler, _in_dim, _resolved_version
    if _model is not None: return
    if MODEL_VERSION:
        base = f"{PREFIX}/{MODEL_VERSION}"
        _resolved_version = MODEL_VERSION
    else:
        mf = read_json_s3(BUCKET, "manifests/stock-updown/latest.json")
        base = mf['s3_path']
        _resolved_version = mf['current_version']
    model_bytes = download_s3_to_bytes(BUCKET, f"{base}/model.pt")
    scaler_bytes = download_s3_to_bytes(BUCKET, f"{base}/scaler.pkl")
    _scaler = pickle.loads(scaler_bytes)
    _in_dim = _scaler.mean_.shape[0]
    m = MLP(_in_dim)
    m.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location='cpu'))
    m.eval(); _model = m


def handler(event, context):
    with Timer() as t:
        try:
            body = event.get('body')
            if isinstance(body, str):
                body = json.loads(body)
            ticker = (body or {}).get('ticker','AAPL')
            df = load_history(ticker, days=60)
            _load_artifacts()
            x = prepare_inference(df, _scaler)
            logit = _model(torch.tensor(x, dtype=torch.float32)).item()
            prob = 1/(1+math.exp(-logit))
            label = int(prob > 0.5)
            res = {"ticker": ticker, "prob_up": prob, "will_close_up": label, "model_version": _resolved_version}
            ms = int(t.elapsed*1000)
            # ---------- metrics ----------
            put_metric(NAMESPACE, "PredictionCount", 1, dims={"ModelVersion": _resolved_version, "Ticker": ticker})
            put_metric(NAMESPACE, "LatencyMs", ms, unit="Milliseconds", dims={"ModelVersion": _resolved_version})
            logger.info(json.dumps({"msg":"predict","ticker":ticker,"ms":ms,"prob":prob,"model_version":_resolved_version}))
            return {"statusCode": 200, "body": json.dumps(res)}
        except Exception as e:
            logger.error(json.dumps({"err":str(e),"model_version":_resolved_version}))
            put_metric(NAMESPACE, "ErrorCount", 1, dims={"ModelVersion": _resolved_version or "unknown"})
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}