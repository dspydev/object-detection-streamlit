import streamlit as st
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Detect objects in the image with the specified confidence level using Faster R-CNN
def object_detection(image, confidence):
    bbox, label, conf = cv.detect_common_objects(image, model='faster_rcnn', confidence=confidence)
    return bbox, label, conf

def main():
    st.title("Object Detection")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read uploaded image
        image = cv2.imread(uploaded_file)

        # Check if the image contains cars
        bbox, label, conf = object_detection(image, confidence=0.5)
        if 'car' not in label:
            st.error("The image does not contain cars.")
            return

        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR")

        # Confidence level slider
        confidence = st.slider("Confidence Level", 0.0, 1.0, 0.5, 0.05)

        if st.button("Detect Cars"):
            # Perform object detection
            output = draw_bbox(image, bbox, label, conf)

            # Count the number of cars
            car_count = label.count('car')

            # Display image with detected objects
            st.subheader(f"Cars Detected ({car_count} cars)")
            st.image(output, channels="BGR")

if __name__ == "__main__":
    main()
