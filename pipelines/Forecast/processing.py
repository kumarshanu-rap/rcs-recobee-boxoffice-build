import argparse
import logging
import os
from datetime import datetime, timedelta
from io import StringIO

import boto3
import joblib
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
cols_to_drop = ["Unnamed: 12", "Nandesh_Master_Data", "recobee_id", "imdb_id"]


def count_by_year(df, year=1):
    today = datetime.now()
    year_ago = today - timedelta(days=365 * year)
    count = df[(df["release_date"] >= year_ago) & (df["release_date"] <= today)].shape[
        0
    ]
    return count


def process_genre(genre):
    if genre in ["Action", "Comedy", "Drama", "Crime"]:
        return genre
    return "Others"


def create_features(df):
    df["year_since_release"] = pd.Timestamp.now().year - df["release_date"].dt.year

    df["month_of_year"] = df["release_date"].dt.month

    df["day_of_month"] = df["release_date"].dt.day

    return df


def classify_cast_1(row):
    if row["cast_1_last_3_mean"] > 100000000 and row["cast_1_lifetime_count"] > 12:
        return "Top-Tier"
    if row["cast_1_last_3_mean"] > 60000000 and row["cast_1_lifetime_count"] > 7:
        return "Mid-Tier"
    return "Low-Tier"


def classify_cast_2(row):
    if row["cast_2_last_3_mean"] > 100000000 and row["cast_2_lifetime_count"] > 12:
        return "Top-Tier"
    if row["cast_2_last_3_mean"] > 60000000 and row["cast_2_lifetime_count"] > 7:
        return "Mid-Tier"
    return "Low-Tier"


def classify_cast_3(row):
    if row["cast_3_last_3_mean"] > 100000000 and row["cast_3_lifetime_count"] > 12:
        return "Top-Tier"
    if row["cast_3_last_3_mean"] > 60000000 and row["cast_3_lifetime_count"] > 7:
        return "Mid-Tier"
    return "Low-Tier"


def classify_dir(row):
    if row["dir_last_3_mean"] > 120000000 and row["dir_lifetime_count"] >= 9:
        return "Top-Tier"
    if row["dir_last_3_mean"] > 75000000 and row["dir_lifetime_count"] >= 5:
        return "Mid-Tier"
    return "Low-Tier"


def bin_openingday(df):
    opd_bins = [
        0,
        5000000,
        10000000,
        25000000,
        50000000,
        75000000,
        100000000,
        150000000,
        200000000,
        250000000,
        300000000,
        350000000,
        400000000,
        500000000,
        700000000,
    ]
    rows = pd.cut(df["OpeningDay_adjusted"], bins=opd_bins, include_lowest=True)
    return rows


def create_cast_class_wrt_openingday(
    df, col_var, temp_var="cast", target_variable="OpeningDay_adjusted"
):
    cast_1 = df[col_var].apply(lambda x: x.split(",")[0] if not pd.isna(x) else "")
    cast_2 = df[col_var].apply(
        lambda x: x.split(",")[1] if not pd.isna(x) and len(x.split(",")) > 1 else ""
    )

    temp1 = pd.concat([cast_1, df[["release_date", target_variable]]], axis=1)
    temp3 = temp1

    last_collection = (
        temp3.groupby(col_var)
        .apply(lambda grp: grp.nlargest(1, "release_date"))[target_variable]
        .reset_index()
        .groupby(col_var)
        .mean()[target_variable]
        .reset_index(name="last_collection")
    )
    last_2_mean = (
        temp3.groupby(col_var)
        .apply(lambda grp: grp.nlargest(2, "release_date"))[target_variable]
        .reset_index()
        .groupby(col_var)
        .mean()[target_variable]
        .reset_index(name="last_2_mean")
    )
    last_3_mean = (
        temp3.groupby(col_var)
        .apply(lambda grp: grp.nlargest(3, "release_date"))[target_variable]
        .reset_index()
        .groupby(col_var)
        .mean()[target_variable]
        .reset_index(name="last_3_mean")
    )
    last_5_mean = (
        temp3.groupby(col_var)
        .apply(lambda grp: grp.nlargest(5, "release_date"))[target_variable]
        .reset_index()
        .groupby(col_var)
        .mean()[target_variable]
        .reset_index(name="last_5_mean")
    )
    last_10_mean = (
        temp3.groupby(col_var)
        .apply(lambda grp: grp.nlargest(10, "release_date"))[target_variable]
        .reset_index()
        .groupby(col_var)
        .mean()[target_variable]
        .reset_index(name="last_10_mean")
    )

    lifetime_count = temp3.groupby(col_var).size().reset_index(name="lifetime_count")

    last_1year_count = (
        temp3.groupby(col_var)
        .apply(lambda x: count_by_year(x, 1))
        .reset_index(name="last_1year_count")
    )
    last_2year_count = (
        temp3.groupby(col_var)
        .apply(lambda x: count_by_year(x, 2))
        .reset_index(name="last_2year_count")
    )
    last_3year_count = (
        temp3.groupby(col_var)
        .apply(lambda x: count_by_year(x, 3))
        .reset_index(name="last_3year_count")
    )
    last_5year_count = (
        temp3.groupby(col_var)
        .apply(lambda x: count_by_year(x, 5))
        .reset_index(name="last_5year_count")
    )
    last_10year_count = (
        temp3.groupby(col_var)
        .apply(lambda x: count_by_year(x, 10))
        .reset_index(name="last_10year_count")
    )
    last_20year_count = (
        temp3.groupby(col_var)
        .apply(lambda x: count_by_year(x, 20))
        .reset_index(name="last_20year_count")
    )

    lifetime_count = temp3.groupby(col_var).size().reset_index(name="lifetime_count")

    new_df = pd.concat(
        (
            df_.set_index([col_var])
            for df_ in [
                last_collection,
                last_2_mean,
                last_3_mean,
                last_5_mean,
                last_10_mean,
                last_1year_count,
                last_2year_count,
                last_3year_count,
                last_5year_count,
                last_10year_count,
                last_20year_count,
                lifetime_count,
            ]
        ),
        axis=1,
    ).reset_index()

    return new_df


if __name__ == "__main__":
    logger.info("Extracting arguments")
    parser = argparse.ArgumentParser()

    # df, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train_s3_uri", type=str, default="train.csv")
    args, _ = parser.parse_known_args()

    logger.info(f"SKLearn Version: {sklearn.__version__}")
    logger.info(f"Joblib Version: {joblib.__version__}")

    # fmt: off
    inflation_data = {
        "Year": [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021, 2022, 2023, 2024],
        "Multiplier": [16.57, 14.96, 13.83, 12.74, 11.81, 11.02, 10.32, 9.44,
                       8.72, 8.04,
                    7.26, 6.39, 5.86, 5.33, 4.85, 4.45, 4.13, 3.88, 3.59, 3.49,
                    3.37, 3.26, 3.14, 3.03, 2.86, 2.71, 2.50, 2.34, 2.14, 2.00,
                    1.81, 1.66, 1.54, 1.45, 1.41, 1.37, 1.33, 1.28, 1.23, 1.20,
                    1.12, 1.12, 1.07, 1.03, 1.00]
    }
    # fmt: on

    inflation_df = pd.DataFrame(inflation_data)

    base_dir = "/opt/ml/processing"
    train_path = f"{base_dir}/train"

    # Fetch the train file object from S3
    logger.info(f"Reading Train Data from: {args.train_s3_uri}")
    bucket_name, train_file_key = args.train_s3_uri.replace("s3://", "").split("/", 1)
    train_response = s3.get_object(Bucket=bucket_name, Key=train_file_key)
    train_file_content = train_response["Body"].read().decode("utf-8")
    train_df = pd.read_csv(StringIO(train_file_content))

    # Processing the data
    logger.info("Processing the data.")
    train_df = train_df.drop(columns=cols_to_drop)
    train_df["release_date"] = pd.to_datetime(
        train_df["release_date"], format="%Y-%m-%d"
    )
    recobee_df = train_df.dropna(subset=["box_office_opening_day", "cast"])
    recobee_df = recobee_df.drop(
        columns=[
            "box_office_opening_weekend",
            "youtube_trailer_id_1",
            "youtube_trailer_id_3",
            "youtube_trailer_id_4",
        ]
    )
    recobee_df_final = recobee_df.dropna()
    recobee_df_final["release_year"] = recobee_df_final["release_date"].dt.year
    recobee_inflation = pd.merge(
        recobee_df_final, inflation_df, left_on="release_year", right_on="Year"
    )
    recobee_inflation["OpeningDay_adjusted"] = (
        recobee_inflation["box_office_opening_day"] * recobee_inflation["Multiplier"]
    )
    recobee_inflation["cast_1"] = recobee_inflation["cast"].apply(
        lambda x: x.split(",")[0] if not pd.isna(x) else ""
    )
    recobee_inflation["cast_2"] = recobee_inflation["cast"].apply(
        lambda x: x.split(", ")[1] if not pd.isna(x) and len(x.split(",")) > 1 else ""
    )
    recobee_inflation["cast_3"] = recobee_inflation["cast"].apply(
        lambda x: x.split(", ")[2] if not pd.isna(x) and len(x.split(",")) > 2 else ""
    )
    cast_1_df = create_cast_class_wrt_openingday(recobee_inflation, col_var="cast_1")
    col_name = [
        "last_collection",
        "last_2_mean",
        "last_3_mean",
        "last_5_mean",
        "last_10_mean",
        "last_1year_count",
        "last_2year_count",
        "last_3year_count",
        "last_5year_count",
        "last_10year_count",
        "last_20year_count",
        "lifetime_count",
    ]
    col_dict = {}
    for i in col_name:
        col_dict[i] = "cast_1_" + i

    cast_1_df = cast_1_df.rename(columns=col_dict)
    cast_2_df = create_cast_class_wrt_openingday(recobee_inflation, col_var="cast_2")
    col_dict = {}
    for i in col_name:
        col_dict[i] = "cast_2_" + i

    cast_2_df = cast_2_df.rename(columns=col_dict)
    cast_3_df = create_cast_class_wrt_openingday(recobee_inflation, col_var="cast_3")
    col_dict = {}
    for i in col_name:
        col_dict[i] = "cast_3_" + i

    cast_3_df = cast_3_df.rename(columns=col_dict)
    dir_df = create_cast_class_wrt_openingday(recobee_inflation, col_var="director")
    dir_dict = {}
    for i in col_name:
        dir_dict[i] = "dir_" + i

    dir_df = dir_df.rename(columns=dir_dict)
    recobee_inflation["genre_final"] = recobee_inflation["genre"].apply(
        lambda x: x.split(",")[0] if not pd.isna(x) else ""
    )
    recobee_inflation["genre_final_2"] = recobee_inflation["genre"].apply(
        lambda x: x.split(", ")[1] if not pd.isna(x) and len(x.split(",")) > 1 else ""
    )
    recobee_inflation.loc[recobee_inflation.genre_final_2 == "War", "genre_final_2"] = (
        "Action"
    )
    recobee_inflation["genre_final"] = recobee_inflation["genre_final"].apply(
        process_genre
    )
    recobee_inflation["genre_final_2"] = recobee_inflation["genre_final_2"].apply(
        process_genre
    )

    recobee_inflation = create_features(recobee_inflation)
    recobee_inflation = pd.merge(
        recobee_inflation, cast_1_df, left_on="cast_1", right_on="cast_1"
    )
    recobee_inflation = pd.merge(
        recobee_inflation, cast_2_df, left_on="cast_2", right_on="cast_2"
    )
    recobee_inflation = pd.merge(
        recobee_inflation, cast_3_df, left_on="cast_3", right_on="cast_3"
    )
    recobee_inflation = pd.merge(
        recobee_inflation, dir_df, left_on="director", right_on="director"
    )
    recobee_inflation = recobee_inflation.drop(columns=["genre"])
    recobee_inflation["cast_1_class"] = recobee_inflation.apply(classify_cast_1, axis=1)
    recobee_inflation["cast_2_class"] = recobee_inflation.apply(classify_cast_2, axis=1)
    recobee_inflation["cast_3_class"] = recobee_inflation.apply(classify_cast_3, axis=1)
    recobee_inflation["dir_class"] = recobee_inflation.apply(classify_dir, axis=1)
    recobee_inflation["OpeningDayBins"] = bin_openingday(recobee_inflation)

    col_to_encode = ["cast_1_class", "cast_2_class", "cast_3_class", "dir_class"]

    logger.info("Starting feature scalling and encoding...")
    label_encoder = LabelEncoder()

    for col in col_to_encode:
        recobee_inflation[col + "new"] = label_encoder.fit_transform(
            recobee_inflation[col]
        )
    recobee_inflation = pd.get_dummies(
        recobee_inflation, columns=["genre_final"], prefix="genre", dtype=int
    )
    recobee_inflation = pd.get_dummies(
        recobee_inflation, columns=["genre_final_2"], prefix="genre", dtype=int
    )

    col_to_scale = [
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
    ]
    scaler = StandardScaler()
    scaler.fit(recobee_inflation[col_to_scale])
    recobee_inflation[col_to_scale] = scaler.transform(recobee_inflation[col_to_scale])
    recobee_inflation.duplicated().sum()
    logger.info(f"Shape of the processed data: {recobee_inflation.shape}")

    # Storing the processed file
    export_path = f"{train_path}/train_processed.csv"
    recobee_inflation.to_csv(export_path, index=False)
    logger.info(f"Features CSV dumped into: {export_path}")
