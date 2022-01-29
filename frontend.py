import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.title("Image Classification KOA")
st.header("Knee OsteoArthritis Classification")
st.text("Upload a KNEE .png Image for image classification as 0-4")

uploaded_file = st.file_uploader("Choose a KNEE image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded KNEE.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    #CALL OUR API
    #label = APICALL(image)
    label=1

if label == 0:
    st.write("The KNEE is Healthy")
elif label == 1:
    st.write("The KNEE is Grade 1 (Doubtful)")
elif label == 2:
    st.write("The KNEE is Grade 2 (Minimal)")
elif label == 3:
    st.write("The KNEE is Grade 3 (Moderate)")
else:
    st.write("The KNEE is Grade 4 (Severe)")
