import streamlit as st
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import io

# Detect objects in the image with the specified confidence level using YOLOv3
def object_detection(image, confidence):
    bbox, label, conf = cv.detect_common_objects(image, model='yolov3', confidence=confidence)
    return bbox, label, conf

def main():
    st.title("Object Detection App")
    st.write("Welcome to the Object Detection App! Here are the steps to use the app:")

    # Summary of steps
    st.subheader("Summary of Steps")
    st.markdown("""
    1. Upload an image.
    2. Adjust the confidence level.
    3. Detect objects in the image.
    """)

    # Step 1: Upload image file
    st.subheader("Step 1: Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR")

        # Step 2: Confidence level slider with explanation
        st.subheader("Step 2: Adjust Confidence Level")
        st.markdown(
            """
            The Confidence Level determines the threshold for object detection. It controls the certainty required for an object to be detected. 
            Move the slider to increase or decrease the Confidence Level and adjust the accuracy of object detection. Higher values may result in 
            fewer detections but with higher accuracy.
            """
        )
        confidence = st.slider(
            "Confidence Level",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Move the slider to adjust the confidence level for object detection."
        )

        if st.button("Detect Objects"):
            # Perform object detection
            output = object_detection(image, confidence)

            # Draw bounding boxes around the detected objects
            output_image = draw_bbox(image, output[0], output[1], output[2])

            # Count the number of objects
            object_count = len(output[1])

            # Display image with detected objects
            st.subheader(f"Objects Detected ({object_count} objects)")
            st.image(output_image, channels="BGR")

if __name__ == "__main__":
    main()
