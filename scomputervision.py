from shelpers import backends, detector, models, df_metrics, delete_representations, get_id
import cv2
from cv2 import VideoCapture
from deepface import DeepFace
import os
import matplotlib.pyplot as plt


def register_face(id, upload_image=None, live = False):
    result = {"success": False, "message": "", "id": id, "image_path": ""}

    # define the path to the image_database directory
    img_db_path = os.path.join(os.getcwd(), 'img_db')

    # create the database if it does not exist
    if not os.path.exists(img_db_path):
        os.mkdir(img_db_path)

    # check for duplicate id
    if id + '.jpg' in os.listdir(img_db_path):
        result["message"] = "id already exists"
        return result


    if live:
        # take in image from cam
        cap = VideoCapture(0)

        while True:
            ret, frame = cap.read()
            face = detector(frame)
            if face:
                # crop and save image
                x, y , w, h, _, _ = face[0]['facial_area'].values()
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
                crop_img = frame[y:y+h, x:x+w]

                # Display the image with the detected face
                cv2.imshow('detect face', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    result["message"] = "quit detected. No face detected in the provided image"
                    break

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    # Save face
                    face_path = os.path.join(img_db_path, id + '.jpg')
                    # face_path = "/img_db/"+id+".jpg"
                    cv2.imwrite(face_path, crop_img)
                    print(f" id {id} registered successfully")
                    print(f"image stored in {face_path}")

                    delete_representations()
                    result["success"] = True
                    result["message"] = f"id {id} registered"
                    result["image_path"] = face_path
                    break

            else:
                # if no face detected
                h,w = frame.shape[:2]
                text = 'Adjust face and brightness'
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontsize = 1
                color = (0,255,0)
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, fontsize, thickness)

                x = (w - text_size[0]) // 2
                y = (h + text_size[1]) // 2
                cv2.putText(frame, text, (x, y), font, fontsize, color, thickness)
                # cv2.putText(frame, text, (10, h - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0,), 1)
                # cv2.putText(crop_img, 'Align face', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('detect face', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    result["message"] = "quit detected. No face detected in the provided image"
                    break

        cv2.destroyAllWindows()
        cap.release()
        return result


    # Load the image
    else:
        # check image_path is not None
        if upload_image is None:
            result["message"] = "No image provided"
            print(result["message"])
            return result

        frame = cv2.imread(upload_image)

        # perform face detection
        face = detector(frame)
        if face:
            # crop and save image
            x, y , w, h, _, _ = face[0]['facial_area'].values()
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
            crop_img = frame[y:y+h, x:x+w]

            plt.imshow(crop_img)

            # Display the image with the detected face
            # cv2.imshow('detect face', frame)
            # cv2.waitKey(0)

            # Save face
            face_path = os.path.join(img_db_path, id + '.jpg')
            # face_path = "/img_db/"+id+".jpg"
            cv2.imwrite(face_path, crop_img)
            print(f" id {id} registered successfully")
            print(f"image stored in {face_path}")

            delete_representations()
            result["success"] = True
            result["message"] = f"id {id} registered successfully"
            result["image_path"] = face_path
            return result

        else:
            result["message"] = "No face detected in the provided image"
            print(result["message"])
            cv2.destroyAllWindows()
            return result


# take in the path to the image database
def face_detect(image_path, append_img=False):
    img_db_path = img_db_path = os.path.join(os.getcwd(), 'img_db')
    frame = cv2.imread(image_path)
    face = detector(frame)
    if append_img:
        result={"id":"","message":"","success":False,"image_array":[]}
    else:
        result={"id":"","message":"","success":False}

    dfs = []
    if face is not None:
        # face detection and recognition logic
        x,y,w,h, _, _ = face[0]['facial_area'].values()
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 3)
        crop_img = frame[y:y+h, x:x+w]
        # cv2.imshow('detect face', frame)
        try:
            file_path=None
            df = DeepFace.find(img_path=crop_img, db_path=img_db_path, model_name=models[2],
                                distance_metric=df_metrics[2], enforce_detection=False)
                            #    detector_backend=backends[0])
            print(df)
            dfs.append(df)
            for d in df:
                img_paths = d["identity"].values
                # print(img_paths)
                file_path=img_paths
            print(file_path)
        except Exception as e:
            print(f"Error in DeepFace.find: {e}")
            result["message"]=e
            result["id"]=""
            result["success"]=False


        def get_id(list_list_df):
            print(list_list_df)
            if len(list_list_df)<1:
                return False
            df = list_list_df[0]
            df = df.split("/")[-1]
            name = df.replace(".jpg","")
            return name

        delete_representations()


        user_id = get_id(file_path)
        if user_id:
            result["id"]=user_id
            result["success"]=True
            result["message"]="successful"
            if append_img:
                result["image_array"]=crop_img

        else:
            result["id"]=""
            result["success"]=False
            result["message"]="No match found"
            if append_img:
                result["image_array"]=crop_img
        # print(result)

        return result

        # break



# # face_detect("C:\\Users\\DELL\\Desktop\\pics.jpg",os.getcwd()+"/img_db/")
# #
# '''
# use case of register_face function
# you can either upload an image or take in live image from camera

# register_face(id, image_path='path\to\image')
# or
# by setting live=True to use webcam
# register_face(id, live=True)
# '''

# def face_detect(image_path, db_path):
#     img_arr = cv2.imread(image_path)
#     face_objs = DeepFace.extract_faces(img_path = img_arr,
#         target_size = (224, 224),
#         detector_backend = backends[4],enforce_detection=False)
#     print("gotten here")

#     face = face_objs[0]['face']
#     file_path=False
#     try:
#         dfs = DeepFace.find(img_path=face, db_path=db_path,
#                             model_name="Facenet512", distance_metric="euclidean_l2",
#                             # detector_backend = backends[0],
#                             enforce_detection=False)
#         print("dfs:",dfs)
#         for df in dfs:
#             img_paths = df["identity"].values
#             # print(img_paths)
#             file_path=img_paths
#         print(file_path)
#     except Exception as e:
#         print(f"Error in DeepFace.find: {e}")


#     def get_id(list_list_df):
#         print(list_list_df)
#         if list_list_df==[]:
#             return "empty"
#         df = list_list_df[0]
#         df = df.split("/")[-1]
#         name = df.split(".")[0]
#         return name

#     delete_representations()


#     user_id = get_id(file_path)
#     print(user_id)

#     return user_id