import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import requests
#como no uso API tambien traigo Keras
import tensorflow

st.title("Image Classification KOA")
st.header("Knee OsteoArthritis Classification")
st.text("(c)2022 by the KOA team")
st.text("Upload a KNEE .png/.jpg/.jpeg Image for image classification as 0-4")

uploaded_file = st.file_uploader("Choose a KNEE image...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded KNEE.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    #traigo lo que tenia en la API porque somos vagos y no queremos desarrollar
    # Prepro
    imagen = image.resize((224,224))
    imagen = imagen.convert('RGB')
    # imagen = np.reshape(imagen, (224,224,3), order='F')
    imagen_np = (np.array(imagen)) / 255
    imagen_exp = np.expand_dims(imagen_np, 0)
    #model = tf.keras.models.load_model(path_model, compile=False)
    model = tensorflow.keras.models.load_model('modelo_franco_MobileNet121.h5',
                                               compile=False)
    prediction = model.predict(imagen_exp)
    predicted_value = f"{np.argmax(prediction, axis=1)}"
    label = predicted_value[1]

    #return {"prediction": predicted_value[1]}
    #CALL OUR API
    #KOA_api_url = f"http://127.0.0.1:8000/predict?image_to_predict={image}"
    #response = requests.get(KOA_api_url).json()
    #label = response['prediction']

    if label == '0':
        st.write("The KNEE is Healthy")
    elif label == '1':
        st.write("The KNEE is Grade 1 (Doubtful)")
    elif label == '2':
        st.write("The KNEE is Grade 2 (Minimal)")
    elif label == '3':
        st.write("The KNEE is Grade 3 (Moderate)")
    else:
        st.write("The KNEE is Grade 4 (Severe)")
