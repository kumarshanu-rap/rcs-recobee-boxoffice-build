import argparse
import json
import logging
import os
import tarfile

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
train_cols = [
    "year_since_release",
    "month_of_year",
    "day_of_month",
    "cast_1_last_2year_count",
    "cast_1_last_3year_count",
    "cast_1_last_5year_count",
    "cast_1_lifetime_count",
    "cast_2_last_2year_count",
    "cast_2_last_3year_count",
    "cast_2_last_5year_count",
    "cast_2_lifetime_count",
    "cast_3_last_2year_count",
    "cast_3_last_3year_count",
    "cast_3_last_5year_count",
    "cast_3_lifetime_count",
    "dir_last_2year_count",
    "dir_last_3year_count",
    "dir_last_5year_count",
    "dir_last_10year_count",
    "dir_last_20year_count",
    "dir_lifetime_count",
    "cast_1_last_collection",
    "cast_1_last_2_mean",
    "cast_1_last_3_mean",
    "cast_1_last_5_mean",
    "cast_1_last_10_mean",
    "cast_2_last_collection",
    "cast_2_last_2_mean",
    "cast_2_last_3_mean",
    "cast_2_last_5_mean",
    "cast_2_last_10_mean",
    "cast_3_last_collection",
    "cast_3_last_2_mean",
    "cast_3_last_3_mean",
    "cast_3_last_5_mean",
    "cast_3_last_10_mean",
    "dir_last_collection",
    "dir_last_2_mean",
    "dir_last_3_mean",
    "dir_last_5_mean",
    "dir_last_10_mean",
    "genre_Action",
    "genre_Crime",
    "genre_Comedy",
    "genre_Drama",
    "genre_Others",
    "genre_Action.1",
    "genre_Crime.1",
    "genre_Comedy.1",
    "genre_Drama.1",
    "genre_Others.1",
]
target_var = "OpeningDayBins"

if __name__ == "__main__":
    logger.info("Extracting arguments")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--test_file", type=str, default="test.csv")
    args, _ = parser.parse_known_args()

    logger.info(f"SKLearn Version: {sklearn.__version__}")
    logger.info(f"Joblib Version: {joblib.__version__}")

    base_dir = "/opt/ml/processing"
    model_dir = f"{base_dir}/model"

    model_path = f"{model_dir}/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall(path=".")
    print("Extracted files:", os.listdir("."))

    # logger.info(f"Listing {args.model_dir}: {os.listdir(args.model_dir)}")
    logger.info(f"Listing {base_dir}: {os.listdir(base_dir)}")
    logger.info(f"Listing {model_dir}: {os.listdir(model_dir)}")

    test_path = "./test.csv"
    logger.info(f"Reading test Data from: {test_path}")

    df = pd.read_csv(test_path)
    logger.info(f"Shape of Test Data: {df.shape}")
    logger.info(f"Columns: {df.columns}")

    logger.info(f"Validation Evaluation started.....")

    X_test = df[train_cols]
    y_test = df[target_var]

    model = joblib.load(os.path.join("./", "model.joblib"))

    logger.info("Performing predictions...")
    y_pred = model.predict(X_test)

    logger.info("Evaluating model...")
    logger.info(f"Classification Report: \n {classification_report(y_pred, y_test)}")

    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info("Best Model Test Accuracy: %s", str(test_accuracy))

    metrics_dir = f"{base_dir}/evaluation"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "evaluation.json")

    evaluation_metrics = {
        "classification_metrics": {
            "accuracy": {"value": test_accuracy, "standard_deviation": "NaN"}
        }
    }

    logger.info(evaluation_metrics)

    with open(metrics_path, "w") as f:
        json.dump(evaluation_metrics, f)
    logger.info(f"Evaluation metrics saved to: {metrics_path}")
