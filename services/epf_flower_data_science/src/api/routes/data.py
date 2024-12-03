from fastapi import APIRouter, HTTPException
from src.services.env import load_environment
import pandas as pd
from src.services.data import train_classification_model
from src.services.cleaning import preprocess_iris_dataset, split_iris_dataset
import os
import subprocess

load_environment()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY")

router = APIRouter()

@router.get("/data/download", name="Download Iris dataset")
def download_iris_dataset():
    """Download and save the Iris dataset to the data folder."""
    dataset_name = "uciml/iris"
    save_path = os.path.join(os.path.dirname(__file__), "../../data")

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_API_KEY")
    
    if not kaggle_username or not kaggle_key:
        raise HTTPException(
            status_code=500,
            detail="Kaggle credentials are not configured properly. Check your .env file."
        )

    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)
    with open(kaggle_json_path, "w") as f:
        f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')

    try:
        os.makedirs(save_path, exist_ok=True)

        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", save_path, "--unzip"],
            check=True
        )

        return {"message": f"Dataset downloaded and saved to {save_path}"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download dataset: {str(e)}"
        )

    finally:
        if os.path.exists(kaggle_json_path):
            os.remove(kaggle_json_path)

import pandas as pd

@router.get("/data/load", name="Load Iris dataset")
def load_iris_dataset():
    """Load the Iris dataset as a DataFrame and return it as JSON."""
    data_folder = os.path.join(os.path.dirname(__file__), "../../data")
    dataset_file = os.path.join(data_folder, "Iris.csv")  

    if not os.path.exists(dataset_file):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found in {data_folder}. Make sure to download it first."
        )

    try:
        df = pd.read_csv(dataset_file)

        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset: {str(e)}"
        )



@router.get("/data/process", name="Process Iris dataset")
def process_iris_dataset():
    """Process the Iris dataset and return the preprocessed DataFrame as JSON."""
    data_folder = os.path.join(os.path.dirname(__file__), "../../data")
    dataset_file = os.path.join(data_folder, "Iris.csv")

    if not os.path.exists(dataset_file):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found in {data_folder}. Make sure to download it first."
        )

    try:
        df = pd.read_csv(dataset_file)

        processed_df = preprocess_iris_dataset(df)

        return processed_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process dataset: {str(e)}"
        )


@router.get("/data/split", name="Split Iris dataset")
def split_iris_dataset_endpoint():
    """Split the Iris dataset into train and test sets and return them as JSON."""
    data_folder = os.path.join(os.path.dirname(__file__), "../../data")
    dataset_file = os.path.join(data_folder, "Iris.csv")

    if not os.path.exists(dataset_file):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found in {data_folder}. Make sure to download it first."
        )

    try:
        df = pd.read_csv(dataset_file)
        processed_df = preprocess_iris_dataset(df)
        train_df, test_df = split_iris_dataset(processed_df)

        return {
            "train": train_df.to_dict(orient="records"),
            "test": test_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to split dataset: {str(e)}"
        )



@router.post("/data/train", name="Train classification model")
def train_model():
    """Train a classification model with the processed dataset."""
    data_folder = os.path.join(os.path.dirname(__file__), "../../data")
    dataset_file = os.path.join(data_folder, "Iris.csv")
    model_config_path = os.path.join(os.path.dirname(__file__), "../../config/model_parameters.json")
    model_save_path = os.path.join(os.path.dirname(__file__), "../../models")

    if not os.path.exists(dataset_file):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found in {data_folder}. Make sure to download it first."
        )

    try:
        df = pd.read_csv(dataset_file)
        processed_df = preprocess_iris_dataset(df)
        train_df, _ = split_iris_dataset(processed_df)
        result = train_classification_model(train_df, model_config_path, model_save_path)
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train model: {str(e)}"
        )


from typing import List
from src.schemas.prediction import PredictionInput
from src.services.data import predict_with_model

@router.post("/data/predict", name="Make Predictions")
def make_predictions(input_data: List[PredictionInput]):
    """Make predictions using the trained classification model."""
    model_path = os.path.join(os.path.dirname(__file__), "../../models/LogisticRegression.joblib")

    try:
        input_data_dict = [data.dict() for data in input_data]
        predictions = predict_with_model(input_data_dict, model_path)
        return {"predictions": predictions}
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to make predictions: {str(e)}"
        )
        
from src.services.firestore import create_firestore_collection

@router.post("/firestore/init", name="Initialize Firestore Collection")
def init_firestore_collection():
    """Create the Firestore collection 'parameters'."""
    try:
        create_firestore_collection()
        return {"message": "Firestore collection 'parameters' initialized successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Firestore collection: {str(e)}"
        )
