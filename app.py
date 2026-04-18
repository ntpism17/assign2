import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Setting the web application
st.set_page_config(page_title="Image Classifier", page_icon="📸")
st.title("📸 AI Image Classifier (MobileNetV2)")
st.write("Upload an image (.jpg, .jpeg, .png) and the AI will predict what it is!")

# 2. setting cache
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

# (1) PLEASE load model to model HERE !


# 3. upload file to streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # open and upoad file
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    st.write("🔄 Classifying...")

    # 4. Image preparing
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 5. Prediction
    # Make the prediction
    # (1) PLEASE source code HERE !
    # (2) PLEASE source code HERE !


    # 6. Show the result
    st.subheader("Predictions:")
    for i, pred in enumerate(top_preds):
        class_name = pred[1].replace('_', ' ').title() # setting the font format
        confidence = pred[2] * 100

        # show class
        st.write(f"**{i+1}. {class_name}** — {confidence:.2f}%")
        st.progress(int(confidence))
