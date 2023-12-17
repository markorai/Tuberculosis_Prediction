import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):

    # Resize image to (64, 64) and convert to grayscale
    image = image.convert('L').resize((64, 64))

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = image_array.astype(np.float32) / 255.0  # Assuming the model requires values between 0 and 1

    # Expand dimensions to match the model's input shape
    data = np.expand_dims(normalized_image_array, axis=0)
    data = np.expand_dims(data, axis=-1)  # Adding channel dimension (1 channel for grayscale)

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

