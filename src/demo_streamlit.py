import base64
import cv2
import torch
import requests
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from utils import save_upload_file, delete_file
from ultralytics import YOLOv10

@st.cache_data(max_entries=1000)
def process_and_display_image(image_path):
    TRAINED_MODEL_PATH = 'models/best.pt'
    model = YOLOv10(TRAINED_MODEL_PATH)

    CONF_THRESHOLD = 0.3
    IMG_SIZE = 640 

    if image_path.startswith(('http://', 'https://')):
        # Load image from URL
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # Load image from local file
        img = Image.open(image_path).convert('RGB')

    img = np.array(img)

    # Predict
    results = model.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
    annotated_img = results[0].plot()

    # Convert annotated image to PIL format
    annotated_img_pil = Image.fromarray(annotated_img)

    # Display image using Streamlit
    st.markdown('**Detection result**')
    st.image(annotated_img_pil, caption='Result Image', use_column_width=True)

def main():
    st.set_page_config(
        page_title="Helmet Detection with YOLOv10",
        page_icon='static/aivn_favicon.png',
        layout="wide"
    )

    col1, _ = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title('Helmet Detection with YOLOv10')
        st.title(':sparkles: :blue[YOLOv10] Helmet Safety Detection Demo')
        
    uploaded_img = st.file_uploader('__Input your image__', type=['jpg', 'jpeg', 'png'])
    example_button = st.button('Run example')

    st.divider()

    if example_button:
        process_and_display_image('static/example_img.jpg')

    if uploaded_img:
        uploaded_img_path = save_upload_file(uploaded_img)
        try:
            process_and_display_image(uploaded_img_path)
        finally:
            delete_file(uploaded_img_path)

if __name__ == '__main__':
    main()
