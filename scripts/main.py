# main.py
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace

from facecore import (
    find_match_from_face_crop,
    register_from_image,
    list_registered_users,
    THRESHOLD,
    DETECTOR_BACKEND,   # from facecore
)

st.set_page_config(page_title="One-Shot Face Recognition", layout="wide")

st.title("🔹 One-Shot Face Recognition System")
st.write("_Live face recognition with one-shot registration_")

tab1, tab2 = st.tabs(["🎥 Live Recognition (Auto)", "➕ Register New Face"])

# ==============================
# TAB 1: LIVE RECOGNITION
# ==============================
with tab1:
    st.subheader("🎥 Live Camera Recognition (automatic)")

    st.markdown(
        """
        - Stand in front of the camera.  
        - When a face is detected, the system will **automatically** try to recognize it.  
        - Bounding box + label will appear on the video.  
        """
    )

    class FaceRecTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_text = "No face"
            self.last_color = (0, 0, 255)
            self.last_box = None  # (x, y, w, h)
            self.frame_count = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # Only do heavy work every 3rd frame
            if self.frame_count % 3 == 0:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                faces = DeepFace.extract_faces(
                    img_path=img_rgb,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                )

                if faces:
                    # Take the first detected face
                    f0 = faces[0]
                    area = f0.get("facial_area", {})
                    x = int(area.get("x", 0))
                    y = int(area.get("y", 0))
                    w = int(area.get("w", 0))
                    h = int(area.get("h", 0))

                    face_rgb = np.asarray(f0["face"])
                    if face_rgb.max() <= 1.0:
                        face_rgb = (face_rgb * 255).astype(np.uint8)
                    else:
                        face_rgb = face_rgb.astype(np.uint8)

                    crop_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

                    # 👉 Only run recognition because we already detected a face
                    result = find_match_from_face_crop(
                        crop_bgr, top_k=1, threshold=THRESHOLD
                    )

                    if result["success"] and result["matches"]:
                        best = result["matches"][0]
                        self.last_text = f"{best['id']} ({best['sim']:.2f})"
                        self.last_color = (0, 255, 0)
                    else:
                        self.last_text = "No match"
                        self.last_color = (0, 0, 255)

                    self.last_box = (x, y, w, h)
                else:
                    self.last_text = "No face"
                    self.last_color = (0, 0, 255)
                    self.last_box = None

            # Draw last known box + label
            if self.last_box is not None:
                x, y, w, h = self.last_box
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    self.last_color,
                    2,
                )

            cv2.putText(
                img,
                self.last_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.last_color,
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="face-live",
        video_processor_factory=FaceRecTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

# ==============================
# TAB 2: REGISTER FACE
# ==============================
with tab2:
    st.subheader("➕ Register a New User")

    person_name = st.text_input("Name / ID of person:")

    snap2 = st.camera_input("Capture face to register")

    if st.button("📌 Register Face"):
        if not person_name.strip():
            st.error("❗ Enter a name first!")
        elif snap2 is None:
            st.error("❗ Capture an image first!")
        else:
            bytes_data = snap2.getvalue()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name

            try:
                reg_result = register_from_image(person_name.strip(), tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if reg_result["success"]:
                st.success(
                    f"✅ Face registered successfully for **{reg_result['id']}**"
                )
                st.image(
                    reg_result["image_path"],
                    caption="Stored face crop",
                    width=250,
                )
            else:
                st.error(f"❌ {reg_result['message']}")

    st.write("---")
    st.write("### 👥 Registered Users")
    users = list_registered_users()
    if users:
        st.json(users)
    else:
        st.info("No users registered yet.")
