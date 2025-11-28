from pydantic import BaseModel

class TrainRequest(BaseModel):
    dataset_name: str