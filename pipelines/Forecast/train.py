import argparse
import io
import json
import logging
import os
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------ SageMaker Inference Functions ------------------
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    # le_storetype = joblib.load(os.path.join(model_dir, "le_storetype.joblib"))
    # le_assortment = joblib.load(os.path.join(model_dir, "le_assortment.joblib"))
    return {
        "model": model,
        # "le_storetype": le_storetype,
        # "le_assortment": le_assortment,
    }


def input_fn(request_body, content_type="text/csv"):
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body))
        logger.info(f"Got the DF with Shape: {df.shape}, Columns: {df.columns}")

        # Load encoders
        # le_storetype = joblib.load("/opt/ml/model/le_storetype.joblib")
        # le_assortment = joblib.load("/opt/ml/model/le_assortment.joblib")

        # df["StoreType"] = le_storetype.transform(df["StoreType"])
        # df["Assortment"] = le_assortment.transform(df["Assortment"])

        df.sort_values(["Store", "Date"], inplace=True)
        df["Sales_Lag1"] = df.groupby("Store")["Sales"].shift(1)
        df["Sales_MA7"] = df.groupby("Store")["Sales"].transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
        df["Sales_MA30"] = df.groupby("Store")["Sales"].transform(
            lambda x: x.shift(1).rolling(30).mean()
        )

        logger.info("Handling missing values dynamically...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["float64", "int64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(0, inplace=True)
        features = [
            "Store",
            "DayOfWeek",
            "Month",
            "Year",
            "StoreType",
            "Assortment",
            "Sales_Lag1",
            "Sales_MA7",
            "Sales_MA30",
        ]
        df = df[features]
        logger.info(f"Inference model with Columns({len(df.columns)}): {df.columns}")
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_artifacts):
    model = model_artifacts["model"]
    logger.info(input_data)
    prediction = model.predict(input_data)
    logger.info(f"Prediction: {prediction}")
    return prediction


def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


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

# ------------------ Main Training Script ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train_file", type=str, default="train_processed.csv")
    parser.add_argument("--n_estimator", type=int, default=15)

    args = parser.parse_args()

    logger.info(f"SKLearn Version: {sklearn.__version__}")
    logger.info(f"Joblib Version: {joblib.__version__}")
    logger.info(f"Considering n_estimator: {args.n_estimator}")

    logger.info("Loading datasets...")
    train = pd.read_csv(os.path.join(args.train, args.train_file))
    df = train[train_cols]

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns}")

    logger.info("ML Model building.....")
    X = df.copy()
    y = train[target_var]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    t_pred = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    train_accuracy = accuracy_score(y_train, t_pred)
    logger.info("Train Accuracy: %s", str(train_accuracy))

    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info("Test Accuracy: %s", str(test_accuracy))

    logger.info("Performing Hyperparameter tuning...")
    param_grid = {
        "bootstrap": [True, False],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "max_features": ["auto", "sqrt", "log2", None],
        "min_samples_leaf": [1, 2, 4, 5, 6, 7, 8],
        "min_samples_split": [2, 5, 10, 15, 20],
        "n_estimators": [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    }
    cv_search = RandomizedSearchCV(
        RandomForestClassifier(),
        param_grid,
        verbose=1,
        scoring="accuracy",
        n_iter=30,
        cv=3,
    )
    cv_search.fit(X_train, y_train)
    logger.info(f"Best Estimator: {cv_search.best_estimator_}")
    rf_random = cv_search.best_estimator_
    y_pred_rand = rf_random.predict(X_test)

    logger.info(
        f"Classification Report: \n {classification_report(y_pred_rand, y_test)}"
    )
    t_pred = rf_random.predict(X_train)
    y_pred = rf_random.predict(X_test)

    train_accuracy = accuracy_score(y_train, t_pred)
    logger.info("Best Model Train Accuracy: %s", str(train_accuracy))

    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info("Best Model Test Accuracy: %s", str(test_accuracy))
    X_test[target_var] = y_test
    test_data = pd.DataFrame(X_test)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(rf_random, os.path.join(args.model_dir, "model.joblib"))
    # joblib.dump(le_storetype, os.path.join(args.model_dir, 'le_storetype.joblib'))
    # joblib.dump(le_assortment, os.path.join(args.model_dir, 'le_assortment.joblib'))
    test_data.to_csv(os.path.join(args.model_dir, "test.csv"), index=False)
    logger.info(f"Saving model to {args.model_dir}...")
