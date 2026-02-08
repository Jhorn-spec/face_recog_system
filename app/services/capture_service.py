import cv2

class CaptureService:
    def __init__(self, device_index=0):
        self.cap = cv2.VideoCapture(device_index)

    def read_frame(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
