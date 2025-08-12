import os
from dotenv import load_dotenv

load_dotenv()

# MLFlow URI
MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "stock-analysis-notebook")
ARTIFACT_LOCATION = os.getenv("ARTIFACT_LOCATION")
MODEL_PREFIX = os.environ.get('MODEL_PREFIX','models/stock-updown')

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
MODEL_VERSION = int(os.getenv("MODEL_VERSION", 0))

# ticker specificities
tickers = os.getenv("TICKERS", "AAPL")
timeline = os.getenv("TIMELINE", "5y")
lookback = int(os.getenv("LOOKBACK", 14))
epochs = int(os.getenv("EPOCHS", 20))
batch_size = int(os.getenv("BATCHSIZE", 64))
lr = float(os.getenv("LR", 0.001))
hidden = int(os.getenv("HIDDEN", 64))
p = float(os.getenv("P", 0.1))
val_friction = float(os.getenv("VAL_FRACTION", 0.2))