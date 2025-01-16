import streamlit as st
import cv2
import numpy as np
from preprocessing import CuneiformProcessor

# Initialize your preprocessing class
processor = CuneiformProcessor()

# Streamlit app title
st.title("Cuneiform Translation App")

# Image upload widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded, process it
if uploaded_file is not None:
    # Convert the uploaded image to a format OpenCV understands
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Show the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image using your custom function
    processed_image = processor.predict(image)

    # Show the processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)

