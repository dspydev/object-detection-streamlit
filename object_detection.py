import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.title("Object Detection with YOLOv8")

# Summary of Steps
st.header("Summary of Steps")
st.write("1. Import an image using the 'Browse Files' button.")
st.write("2. Adjust the confidence threshold and bounding box thickness using the sliders.")
st.write("3. Select the YOLOv8 model using the radio buttons.")
st.write("4. Click on the 'Detect' button to run object detection on the imported image.")

# Step 1: Import an Image
st.header("Step 1: Import an Image")
st.write("Click on the 'Browse Files' button to import an image from your computer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image")
    image = np.array(image)

# Step 2: Adjust the Parameters
st.header("Step 2: Adjust the Parameters")
st.write("Use the sliders below to adjust the confidence threshold and bounding box thickness for object detection.")
st.write("- The 'Confidence threshold' slider allows you to adjust the confidence threshold for object detection. The higher the value, the fewer objects will be detected but with higher accuracy.")
st.write("- The 'Bounding box thickness' slider allows you to adjust the thickness of the bounding boxes around detected objects.")

# define some parameters
CONFIDENCE = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
font_scale = 1
thickness = st.slider("Bounding box thickness", 1, 10, 2)

# Step 3: Select the Model
st.header("Step 3: Select the Model")
st.write("Use the radio buttons below to select the YOLOv8 model you want to use for object detection.")

# add radio buttons for selecting the model
model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
model_selection = st.radio("Select model:", model_options)
model = YOLO(model_selection)

st.subheader("Model Differences")
st.write("- `yolov8n.pt`: This is the default YOLOv8 model that provides a good balance between speed and accuracy.")
st.write("- `yolov8s.pt`: This is a smaller version of the YOLOv8 model that is faster but less accurate than the default model.")
st.write("- `yolov8m.pt`: This is a larger version of the YOLOv8 model that is more accurate but slower than the default model.")

# Step 4: Run Object Detection
st.header("Step 4: Run Object Detection")
st.write("Click on the 'Detect' button to run object detection on the imported image.")

if st.button("Detect"):
    # run inference on the image 
    # see: https://docs.ultralytics.com/modes/predict/#arguments for full list of arguments
    results = model.predict(image, conf=CONFIDENCE)[0]

    # create a PIL Image object from the numpy array
    image = Image.fromarray(image)

    # create a Draw object to draw on the image
    draw = ImageDraw.Draw(image)

    # loop over the detections
    for data in results.boxes.data.tolist():
        # get the bounding box coordinates, confidence, and class id 
        xmin, ymin, xmax, ymax, confidence, class_id = data
        # converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        # draw a bounding box rectangle and label on the image
        color = tuple([int(c) for c in np.random.randint(0,
                                                         255,
                                                         size=(3,),
                                                         dtype="uint8")])
        draw.rectangle(((xmin, ymin), (xmax, ymax)),
                       outline=color,
                       width=thickness)
        text = f"{confidence:.2f}"
        font = ImageFont.truetype("arial.ttf", size=16)
        text_width, text_height = draw.textsize(text, font=font)
        text_offset_x = xmin
        text_offset_y = ymin - text_height - 5
        draw.rectangle(((text_offset_x,
                         text_offset_y),
                        (text_offset_x + text_width + 2,
                         text_offset_y + text_height)),
                       fill=color)
        draw.text((text_offset_x,
                   text_offset_y),
                  text,
                  fill=(0,
                        0,
                        0),
                  font=font)

    st.image(image)
    st.write(f"Number of objects detected: {len(results.boxes.data.tolist())}")

st.info(f"""
**About Yolov8**

**What is Yolov8?**
Yolov8 is an object detection model that offers advantages over older models in terms of speed and accuracy.

**Advantages of Yolov8:**
- Fast and accurate object detection in real-time.

**Real-world use cases for Yolov8:**
- Suitable for applications such as autonomous driving, surveillance systems, and robotics.
- Can be used for tasks such as object tracking and counting.

**Limitations of this Streamlit application:**
- This Streamlit application is very limited in terms of functionality and accuracy due to Streamlit's memory consumption limitations.
- It is only a prototype that can be significantly improved and deployed on other appropriate platforms such as Azure, AWS or Google Cloud Platform.

[Learn more about YOLOv8](https://ultralytics.com/yolov8)
""")
