import os, json, boto3
from datetime import datetime
from train import train_once
from utils import logger, write_json_s3
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

LAMBDA_NAME = os.environ['INFERENCE_FUNCTION_NAME']  # target serving fn
ALIAS_NAME = os.environ.get('ALIAS_NAME','prod')
CANARY_PERCENT = int(os.environ.get('ROUTE_NEW_PERCENT','10'))  # e.g., 10

lam = boto3.client('lambda', )

def _publish_new_version(model_version: str) -> int:
    # Set env MODEL_VERSION=model_version then publish a numbered version
    conf = lam.get_function_configuration(FunctionName=LAMBDA_NAME)
    env = conf.get('Environment',{}).get('Variables',{})
    env['MODEL_VERSION'] = model_version
    lam.update_function_configuration(FunctionName=LAMBDA_NAME, Environment={'Variables': env})
    lam.publish_version(FunctionName=LAMBDA_NAME)
    vers = lam.list_versions_by_function(FunctionName=LAMBDA_NAME)['Versions']
    new_ver = max(int(v['Version']) for v in vers if v['Version'].isdigit())
    return new_ver


def _set_alias_canary(new_ver: int):
    # If alias exists, keep current primary (stable), add canary weight to new_ver
    try:
        alias = lam.get_alias(FunctionName=LAMBDA_NAME, Name=ALIAS_NAME)
        stable_ver = alias['FunctionVersion']
        lam.update_alias(
            FunctionName=LAMBDA_NAME, Name=ALIAS_NAME,
            FunctionVersion=stable_ver,
            RoutingConfig={'AdditionalVersionWeights': {str(new_ver): CANARY_PERCENT/100.0}}
        )
        return int(stable_ver)
    except lam.exceptions.ResourceNotFoundException:
        # First deployment: 100% to new version
        lam.create_alias(FunctionName=LAMBDA_NAME, Name=ALIAS_NAME, FunctionVersion=str(new_ver))
        return new_ver


def handler(event, context):
    # 1) Train and persist artifacts
    model_version, meta = train_once()
    # 2) Publish new serving version with MODEL_VERSION pinned
    new_ver = _publish_new_version(model_version)
    # 3) Route alias to canary
    stable_ver = _set_alias_canary(new_ver)
    # 4) Write deploy state manifest (for dashboards & audits)
    deploy_state = {
        "timestamp": datetime.utcnow().isoformat(),
        "stable_version": stable_ver,
        "canary_version": new_ver,
        "canary_weight_percent": CANARY_PERCENT,
        "model_s3_version": model_version
    }
    write_json_s3(BUCKET, "manifests/stock-updown/deploy_state.json", deploy_state)
    logger.info(json.dumps({"msg":"trained","model_version":model_version,"new_lambda_version":new_ver,"stable":stable_ver}))
    return {"statusCode": 200, "body": json.dumps({"deploy": deploy_state, "meta": meta})}