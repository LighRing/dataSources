import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def train_classification_model(train_df, model_config_path, model_save_path):
    """
    Train a classification model using the train dataset.
    
    Args:
        train_df (pd.DataFrame): Processed training dataset.
        model_config_path (str): Path to the model configuration file.
        model_save_path (str): Path to save the trained model.

    Returns:
        dict: Training accuracy and model details.
    """
    with open(model_config_path, "r") as f:
        config = json.load(f)

    model_type = config["model_type"]
    parameters = config["parameters"]

    X_train = train_df.drop(columns=["Species", "Species_Encoded"])
    y_train = train_df["Species_Encoded"]

    if model_type == "LogisticRegression":
        model = LogisticRegression(**parameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_save_path, f"{model_type}.joblib"))

    return {
        "model_type": model_type,
        "parameters": parameters,
        "training_accuracy": accuracy
    }


def predict_with_model(input_data: list, model_path: str) -> list:
    """
    Make predictions using the trained classification model.
    
    Args:
        input_data (list): A list of dictionaries representing the input features.
        model_path (str): Path to the saved model.

    Returns:
        list: Predictions made by the model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    model = joblib.load(model_path)

    input_df = pd.DataFrame(input_data)

    required_columns = model.feature_names_in_ 
    missing_columns = set(required_columns) - set(input_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    predictions = model.predict(input_df)

    return predictions.tolist()
