import streamlit as st
from scomputervision import register_face
from deepface import DeepFace

st.header("Welcome to Face Recognition Interface")

st.text_input("Enter your name", key="name")
st.text_area("Enter id number, e.g. matric number", key="id")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
# create button at the center
if st.button("Register"):
    if uploaded_file is not None:
        result = register_face(uploaded_file)
        st.write(result)

