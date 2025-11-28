from fastapi import APIRouter
from backend.src.shcemas.input import TrainRequest
from backend.src.core.config import settings

router = APIRouter()

DATASETS = {
    "retail": settings.RETAIL_EVENTS,
    "marketplace": settings.MARKETPLACE_EVENTS,
}

@router.post("/train")
def train(payload: TrainRequest):
    dataset_key = payload.dataset_name
    if dataset_key not in DATASETS:
        return {"error": "Dataset not found"}

    dataset_path = DATASETS[dataset_key]

    return {"status": "training_started", "dataset_path": str(dataset_path)}
