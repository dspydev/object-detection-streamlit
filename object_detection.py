import streamlit as st
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import requests
from urllib.parse import urlparse

# Detect objects in the image with the specified confidence level using the specified model
def object_detection(image, confidence, model):
    # Convert image to float32 before creating blob
    image = image.astype(np.float32)
    
    # Perform object detection using the specified model
    bbox, label, conf = cv.detect_common_objects(image, model=model, confidence=confidence)
    return bbox, label, conf

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def main():
    st.title("Object Detection App")
    st.write("Welcome to the Object Detection App! Here are the steps to use the app:")

    # Summary of steps
    st.subheader("Summary of Steps")
    st.markdown("""
    1. Upload an image or enter an image URL.
    2. Choose an object detection model.
    3. Adjust the confidence level.
    4. Detect objects in the image.
    """)

    # Information about object detection models
    st.subheader("About Object Detection Models")
    st.write("This application allows you to choose between two state-of-the-art object detection models: YOLOv8 and YOLO-NAS.")
    
    st.write("YOLOv8 is a state-of-the-art object detection model that offers high accuracy and fast performance. It is an improvement over previous versions of YOLO in terms of accuracy and speed. You can learn more about YOLOv8 and its capabilities on the [Ultralytics website](https://ultralytics.com/yolov5).")
    
    st.write("YOLO-NAS is a state-of-the-art object detection model that offers high accuracy and fast performance. It outperforms both YOLOv6 & YOLOv8 models in terms of mAP (mean average precision) and inference latency. You can learn more about YOLO-NAS and its capabilities on the [Deci.ai website](https://deci.ai).")

    # Step 1: Upload image file or enter image URL
    st.subheader("Step 1: Upload Image or Enter Image URL")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    image_url = st.text_input('Or enter an image URL')
    
    if uploaded_file is not None:
        # Read uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR")

        # Step 2: Choose object detection model
        st.subheader("Step 2: Choose Object Detection Model")
        model_options = ["YOLOv8", "YOLO-NAS"]
        model_choice = st.selectbox(
            "Object Detection Model",
            model_options,
            help="Choose an object detection model to use."
        )

        # Step 3: Confidence level options with explanation
        st.subheader("Step 3: Adjust Confidence Level")
        st.markdown(
            """
            The Confidence Level determines the threshold for object detection. It controls the certainty required for an object to be detected. 
            Choose an option or enter a custom value to adjust the accuracy of object detection. Higher values may result in fewer detections but with higher accuracy.
            """
        )
        confidence_options = ["Low (0.3)", "Medium (0.5)", "High (0.7)", "Custom"]
        confidence_choice = st.selectbox(
            "Confidence Level",
            confidence_options,
            help="Choose an option or enter a custom value to adjust the confidence level for object detection."
        )
        if confidence_choice == "Custom":
            confidence = st.number_input(
                "Custom Confidence Level",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Enter a custom value for the confidence level for object detection."
            )
        else:
            confidence = float(confidence_choice.split()[1][1:-1])

        if st.button("Detect Objects"):
            # Perform object detection with bounding boxes using the chosen model
            output = object_detection(image, confidence, model=model_choice.lower())

            # Draw rectangular bounding boxes around the detected objects
            output_image = draw_bbox(image.copy(), output[0], output[1], output[2])

            # Count the number of objects
            object_count = len(output[1])

            # Display image with detected objects
            st.subheader(f"Objects Detected ({object_count} objects)")
            st.image(output_image, channels="BGR")
            
    elif is_valid_url(image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            
            # Display original image
            st.subheader("Original Image")
            st.image(img, channels="BGR")

            # Step 2: Choose object detection model
            st.subheader("Step 2: Choose Object Detection Model")
            model_options = ["YOLOv8", "YOLO-NAS"]
            model_choice = st.selectbox(
                "Object Detection Model",
                model_options,
                help="Choose an object detection model to use."
            )

            # Step 3: Confidence level options with explanation
            st.subheader("Step 3: Adjust Confidence Level")
            st.markdown(
                """
                The Confidence Level determines the threshold for object detection. It controls the certainty required for an object to be detected. 
                Choose an option or enter a custom value to adjust the accuracy of object detection. Higher values may result in fewer detections but with higher accuracy.
                """
            )
            confidence_options = ["Low (0.3)", "Medium (0.5)", "High (0.7)", "Custom"]
            confidence_choice = st.selectbox(
                "Confidence Level",
                confidence_options,
                help="Choose an option or enter a custom value to adjust the confidence level for object detection."
            )
            if confidence_choice == "Custom":
                confidence = st.number_input(
                    "Custom Confidence Level",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help="Enter a custom value for the confidence level for object detection."
                )
            else:
                confidence = float(confidence_choice.split()[1][1:-1])

            if st.button("Detect Objects"):
                # Perform object detection with bounding boxes using the chosen model
                output = object_detection(img, confidence, model=model_choice.lower())

                # Draw rectangular bounding boxes around the detected objects
                output_image = draw_bbox(img.copy(), output[0], output[1], output[2])

                # Count the number of objects
                object_count = len(output[1])

                # Display image with detected objects
                st.subheader(f"Objects Detected ({object_count} objects)")
                st.image(output_image, channels="BGR")
                
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
        
    # Information about the developer and GitHub repository
    st.subheader("About the Developer")
    st.write("This application was developed by Damien SOULÉ.")
    st.write("You can find the GitHub repository for this project [here](https://github.com/dspydev/object-detection-streamlit).")
    st.write("You can also follow Damien SOULÉ on [LinkedIn](https://www.linkedin.com/in/damiensoule/).")
    
if __name__ == "__main__":
    main()
