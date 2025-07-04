import argparse
import io
import json
import logging
import os
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument("--n_estimators", type=int, default=100)
    # parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--test_file", type=str, default="test.csv")
    args, _ = parser.parse_known_args()

    logger.info(f"SKLearn Version: {sklearn.__version__}")
    logger.info(f"Joblib Version: {joblib.__version__}")
    logger.info(f"Python Version: {sys.version}")

    base_dir = "/opt/ml/processing"
    model_dir = f"{base_dir}/model"

    model_path = f"{model_dir}/model.tar.gz"
    # model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall(path=".")
    print("Extracted files:", os.listdir("."))

    # logger.info(f"Listing {args.model_dir}: {os.listdir(args.model_dir)}")
    logger.info(f"Listing {base_dir}: {os.listdir(base_dir)}")
    logger.info(f"Listing {model_dir}: {os.listdir(model_dir)}")

    test_path = f"./test.csv"
    logger.info(f"Reading test Data from: {test_path}")

    df = pd.read_csv(test_path)
    logger.info(f"Shape of Test Data: {df.shape}")
    logger.info(f"Columns: {df.columns}")

    logger.info(f"Validation Evaluation started.....")

    features = ['Store', 'DayOfWeek', 'Month', 'Year', 'StoreType', 'Assortment', 'Sales_Lag1', 'Sales_MA7', 'Sales_MA30']
    target = 'Sales'

    X_test = df[features]
    y_test = df[target]


    model = joblib.load(os.path.join('./', "model.joblib"))

    logger.info("Performing predictions...")
    y_pred = model.predict(X_test)

    logger.info("Evaluating model...")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\u2705 RMSE: {rmse:.2f}")
    logger.info(f"\u2705 MAE: {mae:.2f}")
    logger.info(f"\u2705 R² Score: {r2:.4f}")


    metrics_dir = f'{base_dir}/evaluation'
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'evaluation.json')

    evaluation_metrics = {
        "regression_metrics": {
            "mae": {
                "value": mae,
                "standard_deviation": "NaN"
            },
            "rmse": {
                "value": rmse,
                "standard_deviation": "NaN"
            },
            "r2": {
                "value": r2,
                "standard_deviation": "NaN"
            }
        }
    }

    logger.info(evaluation_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f)
    logger.info(f"Evaluation metrics saved to: {metrics_path}")
