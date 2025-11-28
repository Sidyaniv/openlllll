from pydantic import BaseModel
from typing import Any, Optional

class RecommendationItem(BaseModel):
    product: str
    probability: float
    confidence: str

class CustomerProfile(BaseModel):
    age: Optional[int] = None
    income: Optional[str] = None
    region: Optional[str] = None
    segment: Optional[str] = None
    # Добавьте другие поля профиля по необходимости

class CustomerPrediction(BaseModel):
    customer_id: str
    profile: dict[str, Any]
    recommendations: list[RecommendationItem]

class TrainResponse(BaseModel):
    status: str
    dataset_size: Optional[int] = None
    accuracy: Optional[float] = None
    features_used: Optional[list[str]] = None
    model_type: Optional[str] = None
    training_time: Optional[str] = None
    message: Optional[str] = None

class PredictResponse(BaseModel):
    status: str
    predictions: Optional[list[CustomerPrediction]] = None
    message: Optional[str] = None

class MetricsResponse(BaseModel):
    accuracy: Optional[float] = None
    dataset_size: Optional[int] = None
    model_type: Optional[str] = None
    feature_importance: Optional[dict[str, float]] = None
    product_distribution: Optional[dict[str, int]] = None
    segment_distribution: Optional[dict[int, int]] = None
    # error не включаем в успешный ответ - ошибки обрабатываются через HTTPException

