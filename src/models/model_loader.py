import pickle
from pathlib import Path

class ModelLoader:
    def __init__(self, model_path: Path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def get_model(self):
        return self.model
