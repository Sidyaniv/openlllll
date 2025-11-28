from fastapi import APIRouter, HTTPException
from src.schemas.input import TrainRequest, PredictRequest
from src.schemas.output import TrainResponse, PredictResponse, MetricsResponse
from src.services.training import TrainingService
from src.services.prediction import PredictionService
from src.services.metrics import MetricsService

router = APIRouter()

# Инициализация сервисов
training_service = TrainingService()
prediction_service = PredictionService(training_service)
metrics_service = MetricsService(training_service)

# Обучающий датасет
TRAIN_DATASET = "train"

# Тестовые датасеты (могут быть любые, начинающиеся с "test")
def is_valid_dataset(dataset_name: str) -> bool:
    """Проверка валидности датасета"""
    if dataset_name == TRAIN_DATASET:
        return True
    # Любой датасет, начинающийся с "test" считается валидным тестовым датасетом
    return dataset_name.startswith("test")

@router.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest):
    """Обучение модели на указанном датасете"""
    dataset_key = payload.dataset_name
    # Обучение возможно только на train датасете
    if dataset_key != TRAIN_DATASET:
        raise HTTPException(
            status_code=400, 
            detail=f"Training is only allowed on '{TRAIN_DATASET}' dataset. Got: '{dataset_key}'"
        )
    
    result = training_service.train(dataset_key)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Training failed"))
    
    return TrainResponse(**result)

@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Получение рекомендаций для клиентов"""
    dataset_key = payload.dataset_name
    
    # Проверяем валидность датасета
    if not is_valid_dataset(dataset_key):
        raise HTTPException(
            status_code=404, 
            detail=f"Dataset '{dataset_key}' not found. Use 'train' for training or 'test*' for predictions."
        )
    
    # Модель обучается на train, но предсказания можно делать для любых тестовых датасетов
    # Проверяем, обучена ли модель на train датасете
    if not training_service.is_model_trained(TRAIN_DATASET):
        raise HTTPException(
            status_code=400, 
            detail=f"Model is not trained. Please train the model first using /train endpoint with dataset_name='{TRAIN_DATASET}'."
        )
    
    # Для предсказаний используем train модель, но dataset_name передаем для идентификации данных
    result = prediction_service.predict(dataset_key, payload.customer_ids)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Prediction failed"))
    
    return PredictResponse(**result)

@router.get("/metrics/{dataset_name}", response_model=MetricsResponse)
def get_metrics(dataset_name: str):
    """Получение метрик модели"""
    # Метрики доступны только для train датасета (на котором обучалась модель)
    if dataset_name != TRAIN_DATASET:
        raise HTTPException(
            status_code=400,
            detail=f"Metrics are only available for '{TRAIN_DATASET}' dataset. Model is trained on '{TRAIN_DATASET}'."
        )
    
    result = metrics_service.get_metrics(dataset_name)
    
    # Проверяем наличие реальной ошибки (не None)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get metrics"))
    
    # Убираем error из ответа (ошибки обрабатываются через HTTPException, не через поле)
    result.pop("error", None)
    
    # Создаем ответ без поля error
    return MetricsResponse(**result)



