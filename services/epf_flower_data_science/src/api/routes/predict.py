import os
from src.services.env import load_environment

from fastapi import APIRouter, HTTPException
from typing import List
from src.schemas.prediction import PredictionInput
from src.services.data import predict_with_model

load_environment()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY")

router = APIRouter()

@router.post("/data/predict", name="Make Predictions", tags=["Predict"])
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