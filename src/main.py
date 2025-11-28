from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.train import router as train_router

app = FastAPI(title="Bank Model Training API")

# Настройка CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router)
