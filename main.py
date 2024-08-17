import streamlit as st
from scomputervision import register_face
from deepface import DeepFace
import tempfile

st.header("Welcome to Face Recognition Interface")

name = st.text_input("Enter your name", key="name")
id = st.text_area("Enter id number, e.g. matric number", key="id")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
st.write(uploaded_file)
# create button at the center

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as file:
        # file.write(uploaded_file.getbuffer())
        temp_path = file.name
    
    if st.button("Register"):
        result = register_face(id, temp_path, live=True)
        st.write(result)


