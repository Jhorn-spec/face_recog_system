import matplotlib.pyplot as plt
from deepface import DeepFace
from cv2 import VideoCapture
import cv2
import os

class AttendanceSystem:
    def __init__(self):
        self.backends = [
            'opencv', 
            'ssd', 
            'dlib', 
            'mtcnn', 
            'retinaface', 
            'mediapipe',
            'yolov8',
            'yunet',
            'fastmtcnn',
            ]

        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.models = [
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
        

    # visualise the frame
    def visulaize_frame(self, frame):
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        plt.imshow(image_rgb)
        plt.axis('off')  # Turn off axis labels
        plt.show()

    # get the id
    def get_id(self, list_list_df):
        df = list_list_df[0][0]
        df['identity'] = df['identity'].apply(lambda x: x.split("/")[-1])
        name = df.iloc[0][0]
        return name


                    
    def detector(self, frame, enforce=True):
        try:
            return DeepFace.extract_faces(frame, detector_backend=self.backends[0], enforce_detection=enforce)
        except ValueError:
            return None


    def delete_representations(self):
        for obj in os.listdir('../images/img_db/'):
            if obj.startswith('repr'):
                path = os.path.join('../images/img_db/', obj)
                os.remove(path)
        print('previous representations deleted')
        return

    def face_detect(self):
        cap = cv2.VideoCapture(0)
        dfs = []
        
        while True:
            ret, frame = cap.read()

            if ret == False:
                break
            
            face = detector(frame)
            if face is not None:
                # face detection and recognition logic
                x,y,w,h = face[0]['facial_area'].values()  
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 3)
                # cv2.imshow('detect face', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    df = DeepFace.find(img_path=frame, db_path="../images/img_db/", model_name="Facenet512", distance_metric="euclidean_l2", enforce_detection=False)
                    dfs.append(df)
                    break
0
            cv2.imshow('display', frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

        user_id = self.get_id(dfs)
        
        return user_id



    def register_face(self, id):
        cap = VideoCapture(0)

        while True:
            ret, frame = cap.read()
            face = self.detector(frame)
            if face:
                # crop and save image
                x,y,w,h = face[0]['facial_area'].values()
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
                crop_img = frame[y:y+h, x:x+w]
                cv2.imshow('detect face', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif cv2.waitKey(1) & 0xFF == ord('c'):
                    # save face
                    face_path = "../images/img_db/"+id+".jpg"
                    cv2.imwrite(face_path, crop_img)
                    print(f" id {id} registered successfully")
                    print(f"image stored in {face_path}")
                    self.delete_representations()
                    
                    break
            else:
                # if no face detected
                h,w = frame.shape[:2]
                text = 'Adjust face and brightness'
                cv2.putText(frame, text, (10, h - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0,), 1)
                # cv2.putText(crop_img, 'Align face', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Register face', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()    