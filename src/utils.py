import json, os, io, logging, sys, time
from typing import Any, Dict
import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# JSON logging for CloudWatch
logger = logging.getLogger("stock_model")
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(h)

# define s3 Client
s3 = boto3.client("s3", region_name=AWS_REGION)

# Read JSON data from S3
def read_json_s3(bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj['Body'].read().decode('utf-8'))

# Write JSON data to S3
def write_json_s3(bucket: str, key: str, payload: Dict[str, Any]):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode('utf-8'))

# Get s3 file
def download_s3_to_bytes(bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read()

# Upload s3 file
def upload_bytes_to_s3(bucket: str, key: str, data: bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=data)

# define timer used throughout code
class Timer:
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0