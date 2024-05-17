import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from PIL import Image
import asyncio
import streamlit_image_zoom
import yaml
from pathlib import Path
import cv2

from image_utils import load_images, draw_polygons, draw_boxes
from model_utils import process_image, filter_mislabeled_images
from ui_components import add_logos, main_ui

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Add logos and main UI
add_logos()
st.text("")
main_ui()

# Initialize the default model loaded flag
default_model_loaded = False

# Select model
uploaded_model = st.file_uploader("Upload your YOLOv8 model file", type=["pt"], key="model_uploader", accept_multiple_files=False)
model_paths = []
default_model_path = Path(config["default_model_path"])
if uploaded_model:
    uploaded_model_content = uploaded_model.read()
    for i in range(16):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model_file:
            tmp_model_file.write(uploaded_model_content)
            model_paths.append(tmp_model_file.name)
else:
    for i in range(16):
        model_paths.append(default_model_path)
    default_model_loaded = True

# Display a message if the default model is loaded
if default_model_loaded:
    st.info(f"Default model loaded from: {default_model_path}")
else:
    st.success("Custom model uploaded successfully.")

# Select dataset directory
dataset_dir = Path(st.text_input("Enter the path to your dataset directory", config["dataset_directory"]))

# Input for percentage of the dataset to process
percentage = st.slider("Percentage of the dataset to process", 1, 100, config["percentage"])

# Add a slider for confidence threshold
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, config["confidence_threshold"], step=0.01)

# Add a checkbox for half precision
use_half = st.checkbox("Use Half Precision", value=config["use_half_precision"])

# Add an input for image size
img_size = st.number_input("Image Size", value=config["image_size"], step=32, min_value=32)

# Set the initial IoU threshold to 0.5 for the first run
initial_iou_threshold = config["initial_iou_threshold"]

# Add a button to start processing images
if st.button("Process Images"):
    if model_paths and dataset_dir:
        dataset_splits = config["dataset_splits"]
        mislabeled_images = []

        total_images = 0
        for split in dataset_splits:
            total_images += len(load_images(dataset_dir / split / 'images'))

        num_images_to_process = int(total_images * (percentage / 100))

        progress_bar = st.progress(0)

        models = [YOLO(model_path).to('cuda') for model_path in model_paths]

        all_images = []
        for split in dataset_splits:
            all_images.extend(load_images(dataset_dir / split / 'images'))

        if num_images_to_process < total_images:
            all_images = np.random.choice(all_images, num_images_to_process, replace=False).tolist()

        results_queue = asyncio.Queue()

        async def model_worker(model, images, results_queue, iou_threshold, processed_images, conf_threshold, use_half, img_size):
            local_mislabeled_images = []
            tp, fp, fn = 0, 0, 0

            for image_path in images:
                result = await process_image(model, image_path, iou_threshold, conf_threshold, use_half, img_size)
                if result:
                    image_path, _, _, _, _, local_tp, local_fp, local_fn = result  # Unpack all values
                    tp += local_tp
                    fp += local_fp
                    fn += local_fn
                    local_mislabeled_images.append(result)
                await results_queue.put(1)
                processed_images[0] += 1

            return local_mislabeled_images, tp, fp, fn

        async def run_processing():
            images_per_model = len(all_images) // len(models)
            tasks = []
            tp, fp, fn = 0, 0, 0

            for i, model in enumerate(models):
                start_idx = i * images_per_model
                end_idx = start_idx + images_per_model if i != len(models) - 1 else len(all_images)
                tasks.append(model_worker(model, all_images[start_idx:end_idx], results_queue, initial_iou_threshold, processed_images, conf_threshold, use_half, img_size))

            results = await asyncio.gather(*tasks)
            for result in results:
                local_mislabeled_images, local_tp, local_fp, local_fn = result
                mislabeled_images.extend(local_mislabeled_images)
                tp += local_tp
                fp += local_fp
                fn += local_fn

            while processed_images[0] < num_images_to_process:
                await results_queue.get()
                progress_bar.progress(processed_images[0] / num_images_to_process)

            print(f"Total TP: {tp}, FP: {fp}, FN: {fn}")

        processed_images = [0]
        asyncio.run(run_processing())

        progress_bar.empty()

        st.session_state["mislabeled_images"] = mislabeled_images

if "mislabeled_images" in st.session_state:
    display_mode = st.radio("Display Mode", ["Polygons", "Bounding Boxes"])

    iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, initial_iou_threshold, step=0.01, key="post_inference_threshold")

    mislabeled_images = st.session_state["mislabeled_images"]

    filtered_mislabeled_images = filter_mislabeled_images(mislabeled_images, iou_threshold)

    if filtered_mislabeled_images:
        st.write(f"Found {len(filtered_mislabeled_images)} potentially mislabeled images:")
        for img_path, result, labels, label_boxes, suspect_boxes, tp, fp, fn in filtered_mislabeled_images:
            st.write(f"Image: {img_path}")
            st.write(f"TP: {tp}, FP: {fp}, FN: {fn}")  # Add TP, FP, FN counts

            if display_mode == "Polygons":
                image_with_annotations = draw_polygons(img_path, result, labels)
            else:
                image_with_annotations = draw_boxes(img_path, result, label_boxes, suspect_boxes)

            if image_with_annotations is not None:
                image_with_annotations_rgb = cv2.cvtColor(image_with_annotations, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_with_annotations_rgb)

                streamlit_image_zoom.image_zoom(pil_image, mode="scroll", keep_resolution=True, size=(1024, 768), zoom_factor=5, increment=1)
            else:
                st.write(f"Error reading image {img_path}")
    else:
        st.write("No potentially mislabeled images found.")
else:
    st.write("Please upload a model file and enter the dataset directory to start processing images.")
