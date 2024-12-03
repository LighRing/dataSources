"""API Router for Fast API."""
from fastapi import APIRouter
from src.api.routes import hello, data, predict, firestore

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(data.router, tags=["Data"])
router.include_router(predict.router, tags=["Predict"])
router.include_router(firestore.router, tags=["Firestore"])
