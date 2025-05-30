import pandas as pd
import numpy as np
import logging
import joblib, sklearn
import argparse
import os, io, json, sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------ SageMaker Inference Functions ------------------
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    le_storetype = joblib.load(os.path.join(model_dir, "le_storetype.joblib"))
    le_assortment = joblib.load(os.path.join(model_dir, "le_assortment.joblib"))
    return {"model": model, "le_storetype": le_storetype, "le_assortment": le_assortment}

def input_fn(request_body, content_type='text/csv'):
    if content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(request_body))
        logger.info(f"Got the DF with Shape: {df.shape}, Columns: {df.columns}")

        # Load encoders
        le_storetype = joblib.load("/opt/ml/model/le_storetype.joblib")
        le_assortment = joblib.load("/opt/ml/model/le_assortment.joblib")

        df['StoreType'] = le_storetype.transform(df['StoreType'])
        df['Assortment'] = le_assortment.transform(df['Assortment'])

        df.sort_values(['Store', 'Date'], inplace=True)
        df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)
        df['Sales_MA7'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(7).mean())
        df['Sales_MA30'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(30).mean())

        logger.info("Handling missing values dynamically...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        features = ['Store', 'DayOfWeek', 'Month', 'Year', 'StoreType', 'Assortment', 'Sales_Lag1', 'Sales_MA7', 'Sales_MA30']
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

def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

# ------------------ Main Training Script ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train_file', type=str, default="Final_Features.csv")
    parser.add_argument('--n_estimator', type=int, default=15)


    args = parser.parse_args()

    logger.info(f"SKLearn Version: {sklearn.__version__}")
    logger.info(f"Joblib Version: {joblib.__version__}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Considering n_estimator: {args.n_estimator}")

    logger.info("Loading datasets...")
    data = pd.read_csv(os.path.join(args.train, args.train_file))
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data columns: {data.columns}")
    
    logger.info("ML Model building.....")

    le_storetype = LabelEncoder()
    le_assortment = LabelEncoder()
    data['StoreType'] = le_storetype.fit_transform(data['StoreType'])
    data['Assortment'] = le_assortment.fit_transform(data['Assortment'])

    data.sort_values(['Store', 'Date'], inplace=True)
    data['Sales_Lag1'] = data.groupby('Store')['Sales'].shift(1)
    data['Sales_MA7'] = data.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(7).mean())
    data['Sales_MA30'] = data.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(30).mean())

    data.dropna(inplace=True)

    features = ['Store', 'DayOfWeek', 'Month', 'Year', 'StoreType', 'Assortment', 'Sales_Lag1', 'Sales_MA7', 'Sales_MA30']
    target = 'Sales'

    train_data = data[data['Date'] < '2015-07-01']
    test_data = data[data['Date'] >= '2015-07-01']

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    logger.info(f"Training model with Columns({len(X_train.columns)}): {X_train.columns}")
    logger.info(f"Training RandomForest model with: {args.n_estimator} n_estimator...")
    model = RandomForestRegressor(n_estimators=args.n_estimator, random_state=42)
    model.fit(X_train, y_train)

    logger.info("Performing predictions...")
    y_pred = model.predict(X_test)

    logger.info("Evaluating model...")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\u2705 RMSE: {rmse:.2f}")
    logger.info(f"\u2705 MAE: {mae:.2f}")
    logger.info(f"\u2705 R² Score: {r2:.4f}")


    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    joblib.dump(le_storetype, os.path.join(args.model_dir, 'le_storetype.joblib'))
    joblib.dump(le_assortment, os.path.join(args.model_dir, 'le_assortment.joblib'))
    test_data.to_csv(os.path.join(args.model_dir,"test.csv"), index=False)
    logger.info(f"Saving model, encoders and test.csv to {args.model_dir}...")