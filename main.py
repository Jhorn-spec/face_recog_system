import streamlit as st
from scomputervision import register_face, face_detect
from deepface import DeepFace
from shelpers import visulaize_frame
import tempfile
import base64
from io import BytesIO
from PIL import Image
import os
import matplotlib.pyplot as plt

def get_img_base64(ge: Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_img(uploaded_image):
     image = Image.open(uploaded_image)
     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as file:
        temp_path = file.name
        image.save(temp_path)
        return temp_path, image
     

st.header("Welcome to Face Recognition Interface")

name = st.text_input("Enter your name", key="name")
id = st.text_area("Enter id number, e.g. matric number", key="id")

# Load the CSS file
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload  image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

rc, _, lc = st.columns(3)
if rc.button("Register"):
        if uploaded_image is not None:
             path, image = process_img(uploaded_image)
        
        result = register_face(id, path)
        st.markdown(f"<div class='registered'>{result}</div>", unsafe_allow_html=True)
        # st.write(result)
        os.remove(path)

if lc.button("Recognize"):
    if uploaded_image is not None:
         path, image= process_img(uploaded_image)
    
    result = face_detect(path,append_img=True)
    img_path = visulaize_frame(result.pop("image_array"))
    st.markdown(f"<div class='registered'>{result}</div>", unsafe_allow_html=True)
    st.image(img_path)
    # st.write(result)
    os.remove(path)
    os.remove(img_path)





