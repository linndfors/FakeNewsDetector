import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "logs.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_fake_news")

BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)