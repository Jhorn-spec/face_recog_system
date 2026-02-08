import os
import cv2
from deepface import DeepFace
from config import IMG_DB_PATH, FACE_MODEL_NAME, FACE_METRIC
from utils.file_utils import delete_representations

class RecognitionService:
    def __init__(self):
        os.makedirs(IMG_DB_PATH, exist_ok=True)

    def register_face(self, user_id, face_crop):
        img_path = os.path.join(IMG_DB_PATH, f"{user_id}.jpg")

        if os.path.exists(img_path):
            return {"success": False, "message": "ID already registered"}

        cv2.imwrite(img_path, face_crop)
        delete_representations(IMG_DB_PATH)

        return {"success": True, "message": "Face registered", "path": img_path}

    def recognize_face(self, face_crop):
        try:
            result = DeepFace.find(
                img_path=face_crop,
                db_path=IMG_DB_PATH,
                model_name=FACE_MODEL_NAME,
                distance_metric=FACE_METRIC,
                enforce_detection=False
            )
            return result
        except Exception as e:
            return None
