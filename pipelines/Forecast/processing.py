import pandas as pd
import numpy as np
import logging
import joblib
import argparse, sys
import os, io, json
import boto3
from io import StringIO
import sklearn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

if __name__ == "__main__":
    logger.info("Extracting arguments")
    parser = argparse.ArgumentParser()


    # df, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train_s3_uri", type=str, default="soh_data_joined_phase5.csv")
    parser.add_argument("--store_s3_uri", type=str, default="soh_data_joined_phase5.csv")
    args, _ = parser.parse_known_args()

    logger.info(f"SKLearn Version: {sklearn.__version__}")
    logger.info(f"Joblib Version: {joblib.__version__}")
    logger.info(f"Python Version: {sys.version}")

    base_dir = "/opt/ml/processing"
    train_path = f"{base_dir}/train"

    # Fetch the train file object from S3
    logger.info(f"Reading Train Data from: {args.train_s3_uri}")
    bucket_name, train_file_key = args.train_s3_uri.replace("s3://", "").split("/", 1)
    train_response = s3.get_object(Bucket=bucket_name, Key=train_file_key)
    train_file_content = train_response['Body'].read().decode('utf-8')
    train_df = pd.read_csv(StringIO(train_file_content), parse_dates=['Date'])


    # Fetch the store file object from S3
    logger.info(f"Reading Store Data from: {args.store_s3_uri}")
    bucket_name, store_file_key = args.store_s3_uri.replace("s3://", "").split("/", 1)
    store_response = s3.get_object(Bucket=bucket_name, Key=store_file_key)
    store_file_content = store_response['Body'].read().decode('utf-8')
    store_df = pd.read_csv(StringIO(store_file_content))

    data = pd.merge(train_df, store_df, on='Store')
    logger.info(f"Combined data shape: {data.shape}")
    logger.info(f"Combined data columns: {data.columns}")

    logger.info("Handling missing values dynamically...")
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)

    logger.info("Feature engineering...")
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    # Storing the processed file
    export_path = f"{train_path}/Final_Features.csv"
    data.to_csv(export_path, index=False)

    logger.info(f"Features CSV dumped into: {export_path}")
