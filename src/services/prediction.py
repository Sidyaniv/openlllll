"""
Сервис предсказаний - ЗАГЛУШКА
TODO: Заменить на реальную ML логику при интеграции
"""
import random
from typing import Optional, Any
from src.services.training import TrainingService

class PredictionService:
    """
    Заглушка для сервиса предсказаний.
    Возвращает мок-рекомендации для разработки фронтенда.
    """
    
    def __init__(self, training_service: TrainingService):
        self.training_service = training_service
        
    def predict(self, dataset_name: str, customer_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Заглушка генерации рекомендаций
        
        TODO: Заменить на реальное предсказание:
        1. Загрузить обученную модель из artifacts/
        2. Загрузить данные клиентов
        3. Предобработать данные клиентов
        4. Вызвать model.predict() или model.predict_proba()
        5. Вернуть реальные рекомендации с вероятностями
        
        Args:
            dataset_name: Имя датасета ("train" или "test")
            customer_ids: Список ID клиентов (опционально)
            
        Returns:
            Словарь с рекомендациями для клиентов
        """
        # Модель обучается на "train", но предсказания можно делать для любых тестовых датасетов
        # Проверяем, обучена ли модель на train датасете
        if not self.training_service.is_model_trained("train"):
            return {
                "status": "error",
                "message": f"Model is not trained. Please train the model first on 'train' dataset using /train endpoint."
            }
        
        # Генерируем мок-рекомендации
        return self._mock_predictions(dataset_name, customer_ids)
    
    def _mock_predictions(self, dataset_name: str, customer_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Мок-предсказания для разработки фронтенда
        """
        products = [
            "Кредитная карта Premium", "Ипотека", "Автокредит", 
            "Инвестиционный счет", "Страхование", "Сберегательный счет",
            "Дебетовая карта Gold", "Потребительский кредит", "Бизнес-счет"
        ]
        
        if customer_ids is None:
            customer_ids = [f"CUST{str(i).zfill(3)}" for i in range(1, 21)]
        
        predictions = []
        for customer_id in customer_ids[:20]:  # Ограничиваем до 20 клиентов
            customer_predictions = []
            selected_products = random.sample(products, min(5, len(products)))
            
            for product in selected_products:
                prob = round(random.uniform(0.1, 0.99), 4)
                confidence = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
                customer_predictions.append({
                    "product": product,
                    "probability": prob,
                    "confidence": confidence
                })
            
            customer_predictions.sort(key=lambda x: x["probability"], reverse=True)
            
            predictions.append({
                "customer_id": customer_id,
                "profile": {
                    "age": random.randint(25, 65),
                    "income": f"{random.randint(30000, 150000)} руб.",
                    "region": random.choice(["Москва", "СПб", "Новосибирск", "Екатеринбург"]),
                    "client_since": f"{random.randint(1, 10)} лет",
                    "segment": random.choice(["Премиум", "Стандарт", "Новый"]),
                    "products_count": random.randint(1, 5)
                },
                "recommendations": customer_predictions[:5]
            })
        
        return {
            "status": "success",
            "predictions": predictions
        }
