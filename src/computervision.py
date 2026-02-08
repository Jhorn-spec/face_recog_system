# scomputervision.py
import os

import cv2
from cv2 import VideoCapture
from deepface import DeepFace

from helpers import backends, detector, models, df_metrics, delete_representations


def register_face(id, upload_image=None, live=False, cam_index: int = 0):
    result = {"success": False, "message": "", "id": id, "image_path": ""}

    img_db_path = os.path.join(os.getcwd(), "img_db")
    os.makedirs(img_db_path, exist_ok=True)

    if f"{id}.jpg" in os.listdir(img_db_path):
        result["message"] = "id already exists"
        return result

    if live:
        cap = VideoCapture(cam_index)
        if not cap.isOpened():
            result["message"] = "cannot open camera"
            return result

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = detector(frame)
            if faces:
                x, y, w, h, _, _ = faces[0]["facial_area"].values()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                crop_img = frame[y : y + h, x : x + w]

                cv2.imshow("detect face", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    result["message"] = "quit detected. No face registered."
                    break
                if key == ord("c"):
                    face_path = os.path.join(img_db_path, f"{id}.jpg")
                    cv2.imwrite(face_path, crop_img)
                    print(f"id {id} registered successfully")
                    print(f"image stored in {face_path}")

                    delete_representations()
                    result["success"] = True
                    result["message"] = f"id {id} registered"
                    result["image_path"] = face_path
                    break
            else:
                h, w = frame.shape[:2]
                text = "Adjust face and brightness"
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontsize = 1
                color = (0, 255, 0)
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, fontsize, thickness)
                x = (w - text_size[0]) // 2
                y = (h + text_size[1]) // 2
                cv2.putText(frame, text, (x, y), font, fontsize, color, thickness)
                cv2.imshow("detect face", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    result["message"] = "quit detected. No face registered."
                    break

        cv2.destroyAllWindows()
        cap.release()
        return result

    # ----- upload image mode -----
    if upload_image is None:
        result["message"] = "No image provided"
        print(result["message"])
        return result

    frame = cv2.imread(upload_image)
    if frame is None:
        result["message"] = "unable to read image"
        return result

    faces = detector(frame)
    if faces:
        x, y, w, h, _, _ = faces[0]["facial_area"].values()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        crop_img = frame[y : y + h, x : x + w]

        face_path = os.path.join(img_db_path, f"{id}.jpg")
        cv2.imwrite(face_path, crop_img)
        print(f"id {id} registered successfully")
        print(f"image stored in {face_path}")

        delete_representations()
        result["success"] = True
        result["message"] = f"id {id} registered successfully"
        result["image_path"] = face_path
        return result

    result["message"] = "No face detected in the provided image"
    print(result["message"])
    cv2.destroyAllWindows()
    return result


def face_detect(image_path, append_img=False):
    img_db_path = os.path.join(os.getcwd(), "img_db")
    frame = cv2.imread(image_path)

    if frame is None:
        return {"id": "", "message": "unable to read image", "success": False}

    faces = detector(frame)
    if append_img:
        result = {"id": "", "message": "", "success": False, "image_array": []}
    else:
        result = {"id": "", "message": "", "success": False}

    if faces is None or not faces:
        result["message"] = "No face detected"
        if append_img:
            result["image_array"] = frame
        return result

    x, y, w, h, _, _ = faces[0]["facial_area"].values()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    crop_img = frame[y : y + h, x : x + w]

    try:
        df_list = DeepFace.find(
            img_path=crop_img,
            db_path=img_db_path,
            model_name=models[2],
            distance_metric=df_metrics[2],
            enforce_detection=False,
        )

        if isinstance(df_list, list) and df_list:
            df = df_list[0]
        else:
            df = df_list

        if df is None or len(df) == 0:
            delete_representations()
            result["message"] = "No match found"
            if append_img:
                result["image_array"] = crop_img
            return result

        best_row = df.iloc[0]
        file_path = str(best_row["identity"])
        user_id = os.path.splitext(os.path.basename(file_path))[0]

        delete_representations()

        result["id"] = user_id
        result["success"] = True
        result["message"] = "successful"
        if append_img:
            result["image_array"] = crop_img
        return result

    except Exception as e:
        print(f"Error in DeepFace.find: {e}")
        result["message"] = str(e)
        result["id"] = ""
        result["success"] = False
        if append_img:
            result["image_array"] = crop_img
        return result
