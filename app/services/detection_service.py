from deepface import DeepFace

class DetectionService:
    def __init__(self, backend="opencv"):
        self.backend = backend

    def detect_faces(self, frame, enforce=True):
        try:
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=self.backend,
                enforce_detection=enforce
            )
            return faces
        except:
            return None

    def crop_face(self, frame, face):
        x, y, w, h = face["facial_area"].values()
        crop = frame[y:y+h, x:x+w]
        return crop
