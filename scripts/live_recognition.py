# live_recognition.py
import cv2
from facecore import find_match_from_image, THRESHOLD

def live_recognition(cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run recognition
        result = find_match_from_image(frame, top_k=1, threshold=THRESHOLD)

        display_text = "No match"
        if result["success"] and result["matches"]:
            best = result["matches"][0]
            display_text = f"{best['id']} ({best['sim']:.2f})"

        # Draw simple label at top-left
        cv2.putText(
            frame,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if result["success"] else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Face Recognition", frame)

        # automatically processes each frame; quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_recognition(0)
