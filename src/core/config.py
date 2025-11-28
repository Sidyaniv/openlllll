from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    MODEL_PATH: Path = Path("artifacts/recommender.pkl")

    # Базовые пути к данным
    DATA_DIR: Path = Path("dataset")
    # Обучающий датасет
    TRAIN_DATASET: Path = DATA_DIR / "train" / "events"
    # Тестирующий датасет
    TEST_DATASET: Path = DATA_DIR / "test" / "events"

settings = Settings()
