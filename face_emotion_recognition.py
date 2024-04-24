import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace

def main():
    st.title("Face Emotion Recongnization")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert file to opencv format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Resize the image
        max_dimension = 250
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            if height > width:
                img = cv2.resize(img, (int(width * max_dimension / height), max_dimension))
            else:
                img = cv2.resize(img, (max_dimension, int(height * max_dimension / width)))

        # Display the image
        st.image(img, channels="BGR", caption='Uploaded Image', use_column_width=True)

        # Analyze the image
        predictions = DeepFace.analyze(img)

        # Extract prediction values
        dominant_emotion_value = predictions[0]['dominant_emotion']
        dominant_gender_value = predictions[0]['dominant_gender']
        dominant_race_value = predictions[0]['dominant_race']

        # Display prediction values
        st.write("Emotion:", dominant_emotion_value)
        st.write("Gender:", dominant_gender_value)
        st.write("Race:", dominant_race_value)

if __name__ == "__main__":
    main()
