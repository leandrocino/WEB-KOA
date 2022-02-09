import streamlit as st
from PIL import Image as pimage
from IPython.display import Image, display
import numpy as np
import requests
#como no uso API tambien traigo Keras
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

model = tensorflow.keras.models.load_model('modelo_franco_MobileNet121.h5',
                                               compile=False)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tensorflow.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tensorflow.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tensorflow.newaxis]
    heatmap = tensorflow.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)
    return heatmap.numpy()
#'''Create a superimposed visualization'''

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = np.array(img_path)
    #keras.preprocessing.image.load_img(img_path)
    #img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    return superimposed_img
    #display(Image(cam_path))


st.title("RX Knee Classification Project")
img_header = pimage.open('Header Osteo AI.png')
st.image(img_header)
st.header(
    "Knee Osteoarthritis Diagnosis from Plain Radiographs using Deep Learning-Based approach and Attention Maps")
st.write("(c) 2022 by Federico I., Franco S., Roberto C., Leandro C.")
st.write("")
st.subheader("Knee osteoarthritis is defined by degeneration of the knee’s articular cartilage the flexible, slippery material that normally protects bones from joint friction and impact.")
st.subheader("The condition also involves changes to the bone underneath the cartilage and can affect nearby soft tissues.")
st.subheader("Knee osteoarthritis is by far the most common type of arthritis to cause knee pain and often referred to as simply knee arthritis.")
st.subheader("Many other less common types of arthritis can also cause knee pain, including rheumatoid arthritis, pseudogout, and reactive arthritis.")
st.write(" ")
st.write("*Disclaimer*: This model is just an educational project that still does not have the necessary validation to be used as a medical diagnosis tool.")
st.write("**Acknowledgements**")
st.write("Chen, Pingjun (2018), “Knee Osteoarthritis Severity Grading Dataset”, Mendeley Data, V1, doi: 10.17632/56rmx5bjcr.1")
st.write("The dataset is organized from [OAI](https://oai.epi-ucsf.org/datarelease/)")
st.write("Source of Database is available [here](https://data.mendeley.com/datasets/56rmx5bjcr/1)")

st.header('Please upload a RX knee image')
uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = pimage.open(uploaded_file)
    st.image(image, caption='Uploaded RX knee.', use_column_width=False)
    st.write("")
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    #traigo lo que tenia en la API porque somos vagos y no queremos desarrollar
    # Prepro
    imagen = image.resize((224,224))
    imagen = imagen.convert('RGB')
    # imagen = np.reshape(imagen, (224,224,3), order='F')
    imagen_np = (np.array(imagen)) / 255
    imagen_exp = np.expand_dims(imagen_np, 0)
    #model = tf.keras.models.load_model(path_model, compile=False)
    prediction = model.predict(imagen_exp)
    predicted_value = f"{np.argmax(prediction, axis=1)}"
    label = predicted_value[1]

    #return {"prediction": predicted_value[1]}
    #CALL OUR API
    #KOA_api_url = f"http://127.0.0.1:8000/predict?image_to_predict={image}"
    #response = requests.get(KOA_api_url).json()
    #label = response['prediction']

    if label == '0':
        st.title("Result: The AI algorithm shows that the RX has a Grade 0 Osteoarthritis (No osteoarthritis) (Kellgren and Lawrence scale)")
    elif label == '1':
        st.title(
            "Result: The AI algorithm shows that the RX has a Grade 1 Osteoarthritis (Mild) (Kellgren and Lawrence scale)"
        )
    elif label == '2':
        st.title(
            "Result: The AI algorithm shows that the RX has a Grade 2 Osteoarthritis (Minimal) (Kellgren and Lawrence scale)"
        )
    elif label == '3':
        st.title(
            "Result: The AI algorithm shows that the RX has a Grade 3 Osteoarthritis (Moderate) (Kellgren and Lawrence scale)"
        )
    else:
        st.title(
            "Result: The AI algorithm shows that the RX has a Grade 4 Osteoarthritis (Severe) (Kellgren and Lawrence scale)"
        )

    #'''The Grad-CAM algorithm'''

    st.write("*Disclaimer*: This model is just an educational project that still does not have the necessary validation to be used as a medical diagnosis tool.")

    agree = st.checkbox('Check to see the GradCam')

    if agree:
        last_conv_layer_name = "conv5_block4_0_bn"

        # Prepare image
        img_array = imagen_exp

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        col1, col2 = st.columns(2)

        with col1:
            st.header("GradCam view")
            st.image(save_and_display_gradcam(imagen, heatmap), use_column_width=False, caption='Class activation heatmap')

        with col2:
            st.header("Standard RX view")
            st.image(imagen, caption = 'Original Image')

        # time.sleep(10)
        # st.balloons()
