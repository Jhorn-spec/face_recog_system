# helpers.py
import os
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import sqlite3
import argparse


backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "fastmtcnn",
]

df_metrics = ["cosine", "euclidean", "euclidean_l2"]

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]


def check_duplicate_registration(id):
    # TODO: implement as needed
    return False


def visulaize_frame(frame):
    """
    NOTE: function name kept as 'visulaize_frame' to avoid breaking imports.
    Saves frame as RGB image and returns file path.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    path = os.path.join(os.getcwd(), "pick.png")
    plt.imsave(path, image_rgb)
    return path


def get_id(list_list_df):
    if list_list_df == []:
        return "empty"
    df = list_list_df[0][0]["identity"][0].split("/")[-1]
    name = df.split(".")[0]
    return name


def delete_representations():
    db_path = os.path.join(os.getcwd(), "img_db")
    if not os.path.isdir(db_path):
        return
    for obj in os.listdir(db_path):
        if obj.endswith(".pkl"):
            path = os.path.join(db_path, obj)
            os.remove(path)
            print("previous representations deleted")
    return


def delete_id(id):
    print("delete_id called for:", id)
    result = {"message": "", "success": False}
    try:
        db_path = os.path.join(os.getcwd(), "img_db")
        if not os.path.isdir(db_path):
            result["message"] = f"img_db directory not found in {os.getcwd()}"
            return result

        found = False
        for obj in os.listdir(db_path):
            if obj.startswith(id):
                path = os.path.join(db_path, obj)
                os.remove(path)
                print(f"{path} deleted")
                result["message"] = f"{path} deleted"
                result["success"] = True
                found = True

        if not found:
            result["message"] = f"{id} not found in {db_path}"

    except Exception as e:
        print(e)
        result["message"] = str(e)
        result["success"] = False

    return result


def detector(frame, enforce=True):
    try:
        return DeepFace.extract_faces(
            frame, detector_backend=backends[0], enforce_detection=enforce
        )
    except ValueError:
        return None


def detection(frame):
    try:
        dfs = DeepFace.find(
            img_path=frame,
            db_path="../images/img_db/",
            model_name="Facenet512",
            distance_metric="euclidean_l2",
        )
        print("face found")
        return dfs
    except ValueError:
        print("face cannot be detected")
        return None


def write_to_db(id, name, position, phone_number):
    try:
        with sqlite3.connect("database.db") as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    position TEXT,
                    phone_number TEXT
                )
                """
            )
            cur.execute(
                "INSERT INTO users (id, name, position, phone_number) VALUES (?, ?, ?, ?)",
                (id, name, position, phone_number),
            )
            conn.commit()
            print(f"user {id} info written successfully")
    except sqlite3.Error as e:
        print(f"Unable to write to database: {e}")


def get_user_info(id):
    try:
        with sqlite3.connect("database.db") as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ?", (id,))
            row = cur.fetchone()
            return row
    except sqlite3.Error as e:
        print(f"Error: {e}")
        return None


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific functions from the script.")
    parser.add_argument("function", type=str, help="Name of the function to run")
    parser.add_argument("--args", nargs="*", help="Arguments for the function")

    args = parser.parse_args()

    if args.function == "delete_id":
        result = delete_id(*args.args)
        print(result)
    elif args.function == "write_to_db":
        write_to_db(*args.args)
    elif args.function == "get_user_info":
        result = get_user_info(*args.args)
        print(result)
    elif args.function == "detection":
        result = detection(*args.args)
        print(result)
    elif args.function == "delete_representations":
        delete_representations()
    else:
        print(f"Function {args.function} is not recognized.")
