import os, json, time, boto3
from datetime import datetime, timedelta
from utils import logger, write_json_s3

from dotenv import load_dotenv
load_dotenv()

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

CW = boto3.client('cloudwatch', region_name=AWS_REGION)
LAM = boto3.client('lambda', region_name=AWS_REGION)

LAMBDA_NAME = os.getenv('INFERENCE_FUNCTION_NAME')
ALIAS_NAME = os.getenv('ALIAS_NAME','prod')
NAMESPACE = os.getenv('METRIC_NAMESPACE','StockUpDown')
BUCKET = os.getenv('AWS_S3_BUCKET')

# Simple policy: promote canary → stable if in last 60 min:
#  - >= MIN_REQ predictions AND ErrorCount rate < MAX_ERR_RATE and p95 LatencyMs < MAX_P95
MIN_REQ = int(os.getenv('PROMOTE_MIN_REQ','100'))
MAX_ERR_RATE = float(os.getenv('PROMOTE_MAX_ERR_RATE','0.01'))
MAX_P95 = float(os.getenv('PROMOTE_MAX_P95_MS','1200'))
WINDOW_MIN = int(os.getenv('PROMOTE_WINDOW_MIN','60'))


def _metric_sum(name, version, start, end):
    resp = CW.get_metric_statistics(
        Namespace=NAMESPACE,
        MetricName=name,
        Dimensions=[{"Name":"ModelVersion","Value":str(version)}],
        StartTime=start, EndTime=end, Period=60, Statistics=['Sum']
    )
    return sum(dp['Sum'] for dp in resp.get('Datapoints',[]))


def _metric_p95(name, version, start, end):
    resp = CW.get_metric_statistics(
        Namespace=NAMESPACE,
        MetricName=name,
        Dimensions=[{"Name":"ModelVersion","Value":str(version)}],
        StartTime=start, EndTime=end, Period=60, ExtendedStatistics=['p95']
    )
    vals=[dp['ExtendedStatistics']['p95'] for dp in resp.get('Datapoints',[]) if 'ExtendedStatistics' in dp]
    return max(vals) if vals else None


def handler(event, context):
    # Identify alias routing
    alias = LAM.get_alias(FunctionName=LAMBDA_NAME, Name=ALIAS_NAME)
    primary = int(alias['FunctionVersion'])
    weights = alias.get('RoutingConfig',{}).get('AdditionalVersionWeights',{})
    if not weights:
        logger.info(json.dumps({"msg":"no_canary","stable":primary}))
        return {"statusCode":200, "body": json.dumps({"status":"no_canary"})}
    # We assume single canary
    canary_ver = int(next(iter(weights.keys())))

    end = datetime.utcnow(); start = end - timedelta(minutes=WINDOW_MIN)
    total = _metric_sum('PredictionCount', canary_ver, start, end)
    errors = _metric_sum('ErrorCount', canary_ver, start, end)
    p95 = _metric_p95('LatencyMs', canary_ver, start, end) or 0
    err_rate = (errors/total) if total else 1.0

    decision = "hold"
    if total >= MIN_REQ and err_rate <= MAX_ERR_RATE and p95 <= MAX_P95:
        # Promote: set alias primary to canary, clear weights
        LAM.update_alias(FunctionName=LAMBDA_NAME, Name=ALIAS_NAME, FunctionVersion=str(canary_ver))
        decision = "promoted"
    elif err_rate > MAX_ERR_RATE*2 or p95 > MAX_P95*1.5:
        # Rollback: remove canary weight
        LAM.update_alias(FunctionName=LAMBDA_NAME, Name=ALIAS_NAME, FunctionVersion=str(primary))
        decision = "rolled_back"
    # Persist decision
    write_json_s3(BUCKET, "manifests/stock-updown/promotion_decision.json", {
        "time": datetime.utcnow().isoformat(),
        "decision": decision,
        "stable_before": primary,
        "canary": canary_ver,
        "samples": total,
        "err_rate": err_rate,
        "p95_ms": p95
    })
    logger.info(json.dumps({"msg":"promotion_check","decision":decision,"samples":total,"err_rate":err_rate,"p95":p95}))
    return {"statusCode":200, "body": json.dumps({"decision": decision, "samples": total, "err_rate": err_rate, "p95": p95})}