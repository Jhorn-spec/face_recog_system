import cv2
from services.capture_service import CaptureService
from services.detection_service import DetectionService
from services.recognition_service import RecognitionService
from utils.face_utils import extract_identity

cap_service = CaptureService()
detector = DetectionService()
recognizer = RecognitionService()

print("Press 'r' to register face, 'q' to quit, or any other key to detect.")

user_id_for_registration = "john001"   # example

while True:
    frame = cap_service.read_frame()
    if frame is None:
        continue

    faces = detector.detect_faces(frame)
    if not faces:
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    face = faces[0]
    crop = detector.crop_face(frame, face)

    x, y, w, h = face["facial_area"].values()
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Live", frame)
    k = cv2.waitKey(1) & 0xFF

    # Register
    if k == ord('r'):
        out = recognizer.register_face(user_id_for_registration, crop)
        print(out)

    # Recognize
    elif k == ord('d'):
        results = recognizer.recognize_face(crop)
        if results:
            identity = extract_identity(results)
            print("Detected:", identity)

    elif k == ord('q'):
        break

cap_service.release()
cv2.destroyAllWindows()
