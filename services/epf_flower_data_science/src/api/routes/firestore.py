import os
from src.services.env import load_environment

from fastapi import APIRouter, HTTPException
from src.services.firestore import create_firestore_collection

load_environment()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY")

router = APIRouter()

@router.post("/firestore/init", name="Initialize Firestore Collection", tags=["Firestore"])
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

from fastapi import APIRouter, HTTPException
from src.services.firestore import get_firestore_parameters

@router.get("/firestore/parameters", name="Retrieve Firestore Parameters")
def retrieve_firestore_parameters():
    """
    Endpoint to retrieve parameters from the Firestore collection.

    Returns:
        dict: Parameters from the Firestore document.
    """
    try:
        parameters = get_firestore_parameters()
        return {"parameters": parameters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Firestore parameters: {str(e)}")
    
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.services.firestore import update_firestore_parameters, add_firestore_parameters

class AddParametersRequest(BaseModel):
    new_parameters: dict

@router.post("/firestore/parameters", name="Add Firestore Parameters")
def add_parameters(request: AddParametersRequest):
    """
    Endpoint to add or overwrite parameters in Firestore.

    Args:
        request (AddParametersRequest): Parameters to add or overwrite.

    Returns:
        dict: Updated parameters from Firestore.
    """
    try:
        updated_parameters = add_firestore_parameters(request.new_parameters)
        return {"updated_parameters": updated_parameters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add Firestore parameters: {str(e)}")

class UpdateParametersRequest(BaseModel):
    updates: dict

@router.put("/firestore/parameters", name="Update Firestore Parameters")
def update_parameters(request: UpdateParametersRequest):
    """
    Endpoint to update parameters in Firestore.

    Args:
        request (UpdateParametersRequest): Parameters to update.

    Returns:
        dict: Updated parameters from Firestore.
    """
    try:
        updated_parameters = update_firestore_parameters(request.updates)
        return {"updated_parameters": updated_parameters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update Firestore parameters: {str(e)}")
