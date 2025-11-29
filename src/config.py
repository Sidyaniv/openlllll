import torch

# Backend
BACKEND_URL = 'http://localhost:8000'

DEFAULT_DATASET_PATH = r'C:\Users\romae\PycharmProjects\PythonProject7\dataset_banking\dataset\small'

# Гиперпараметры модели
EMB_DIM = 512
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.01
LAMBDA_U = 0.01
LAMBDA_I = 0.001
LAMBDA_J = 0.001
K_RECS = 100  # Топ-K для генерации (backend берёт топ-10)

# Устройство
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Положительные действия
POSITIVE_ACTIONS = ['view', 'click', 'add_to_cart', 'order']