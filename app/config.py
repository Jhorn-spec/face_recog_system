import os

BASE_DIR = os.getcwd()
IMG_DB_PATH = os.path.join(BASE_DIR, "img_db")
DB_PATH = os.path.join(BASE_DIR, "database.db")

FACE_MODEL_NAME = "Facenet512"
FACE_METRIC = "euclidean_l2"
DETECTOR_BACKEND = "opencv"
