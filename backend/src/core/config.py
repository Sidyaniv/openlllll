from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    MODEL_PATH: Path = Path("artifacts/recommender.pkl")

    # Базовые пути к данным
    DATA_DIR: Path = Path("dataset")
    RETAIL_EVENTS: Path = DATA_DIR / "retail/events"
    RETAIL_ITEMS: Path = DATA_DIR / "retail/items.pq"
    MARKETPLACE_EVENTS: Path = DATA_DIR / "marketplace/events"
    MARKETPLACE_ITEMS: Path = DATA_DIR / "marketplace/items.pq"
    USERS: Path = DATA_DIR / "users.pq"
    BRANDS: Path = DATA_DIR / "brands.pq"

settings = Settings()
