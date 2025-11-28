"""
Сервис метрик модели - ЗАГЛУШКА
TODO: Заменить на реальную ML логику при интеграции
"""
import random
from typing import Any
from src.services.training import TrainingService

class MetricsService:
    """
    Заглушка для сервиса метрик.
    Возвращает мок-метрики для разработки фронтенда.
    """
    
    def __init__(self, training_service: TrainingService):
        self.training_service = training_service
        
    def get_metrics(self, dataset_name: str) -> dict[str, Any]:
        """
        Заглушка получения метрик модели
        
        TODO: Заменить на реальные метрики:
        1. Загрузить обученную модель
        2. Вычислить feature_importance из модели
        3. Загрузить реальные данные для анализа распределений
        4. Вычислить реальные метрики качества (accuracy, precision, recall, etc.)
        5. Вернуть реальные метрики
        
        Args:
            dataset_name: Имя датасета ("train" или "test")
            
        Returns:
            Словарь с метриками модели
        """
        # Перед выдачей метрик проверяем, обучена ли модель
        if not self.training_service.is_model_trained(dataset_name):
            return {
                "error": (
                    f"Model for dataset '{dataset_name}' is not trained yet. "
                    "Please train the model via /train endpoint before requesting metrics."
                )
            }
        
        # Проверяем метаданные модели
        metadata = self.training_service.get_model_metadata(dataset_name)
        
        try:
            if metadata:
                # Используем метаданные если есть
                result = {
                    "accuracy": metadata.get("accuracy"),
                    "dataset_size": metadata.get("dataset_size"),
                    "model_type": metadata.get("model_type", "MockModel"),
                    "feature_importance": self._get_feature_importance_from_metadata(metadata),
                    "product_distribution": self._mock_metrics(dataset_name).get("product_distribution"),
                    "segment_distribution": self._mock_metrics(dataset_name).get("segment_distribution")
                }
                # Убеждаемся что error не включен в успешный ответ
                result.pop("error", None)
                return result
            
            # Возвращаем мок-метрики
            result = self._mock_metrics(dataset_name)
            # Убеждаемся что error не включен в успешный ответ
            result.pop("error", None)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def _get_feature_importance_from_metadata(self, metadata: dict[str, Any]) -> dict[str, float]:
        """Получить важность фич из метаданных"""
        features = metadata.get("features_used", [])
        if features:
            # Создаем простую важность на основе порядка фич
            return {feat: round(1.0 / (i + 1), 4) for i, feat in enumerate(features[:10])}
        return self._mock_metrics("").get("feature_importance")
    
    def _mock_metrics(self, dataset_name: str) -> dict[str, Any]:
        """
        Мок-метрики для разработки фронтенда
        """
        feature_importance = {
            "transaction_frequency": round(random.uniform(0.1, 0.3), 4),
            "income_level": round(random.uniform(0.08, 0.25), 4),
            "age": round(random.uniform(0.05, 0.15), 4),
            "credit_score": round(random.uniform(0.07, 0.2), 4),
            "region": round(random.uniform(0.03, 0.1), 4)
        }
        
        product_distribution = {
            "Кредитные карты": random.randint(1000, 5000),
            "Ипотека": random.randint(500, 2000),
            "Автокредиты": random.randint(800, 3000),
            "Сберегательные счета": random.randint(2000, 6000)
        }
        
        segment_distribution = {
            0: random.randint(2000, 5000),
            1: random.randint(1500, 4000),
            2: random.randint(1000, 3000),
            3: random.randint(500, 2000)
        }
        
        return {
            "accuracy": round(random.uniform(0.82, 0.94), 4),
            "dataset_size": random.randint(25000, 75000),
            "model_type": "MockModel",
            "feature_importance": feature_importance,
            "product_distribution": product_distribution,
            "segment_distribution": segment_distribution
        }
