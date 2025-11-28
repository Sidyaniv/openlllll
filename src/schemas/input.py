from pydantic import BaseModel
from typing import Optional

class TrainRequest(BaseModel):
    dataset_name: str

class PredictRequest(BaseModel):
    dataset_name: str
    customer_ids: Optional[list[str]] = None