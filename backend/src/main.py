from fastapi import FastAPI
from backend.src.routes.train import router as train_router

app = FastAPI(title="Bank Model Training API")

app.include_router(train_router, prefix="/api")
