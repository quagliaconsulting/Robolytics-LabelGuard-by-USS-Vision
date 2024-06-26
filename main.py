import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from PIL import Image
import streamlit_image_zoom
import yaml
from pathlib import Path
import cv2
import torch
import multiprocessing as mp
import time

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

# Load data.yaml for class names
dataset_dir = Path(config["dataset_directory"])
with open(dataset_dir / "data.yaml", "r") as data_file:
    data_config = yaml.safe_load(data_file)
class_names = data_config["names"]

# Define the model_worker function
def model_worker(model_path, images, iou_threshold, conf_threshold, use_half, img_size, total_images, device, progress_value, lock):
    model = YOLO(model_path).to(device)
    local_mislabeled_images = []
    tp, fp, fn = 0, 0, 0

    for image_path in images:
        result = process_image(model, image_path, iou_threshold, conf_threshold, use_half, img_size, class_names)
        if result:
            image_path, _, _, _, _, local_tp, local_fp, local_fn = result
            tp += local_tp
            fp += local_fp
            fn += local_fn
            local_mislabeled_images.append(result)
        with lock:
            progress_value.value += 1

    return local_mislabeled_images, tp, fp, fn

# Define the run_processing function
def run_processing(model_paths, all_images, initial_iou_threshold, conf_threshold, use_half, img_size, num_images_to_process, device):
    images_per_model = len(all_images) // len(model_paths)
    tasks = []
    tp, fp, fn = 0, 0, 0

    with mp.Manager() as manager:
        progress_value = manager.Value('i', 0)
        lock = manager.Lock()
        progress_bar = st.progress(0)

        with mp.Pool(processes=config["num_processes"]) as pool:
            # Distribute images among processes
            images_split = np.array_split(all_images, config["num_processes"])

            # Create a list of arguments for each process
            args = [(model_paths[i % len(model_paths)], images_split[i], initial_iou_threshold, conf_threshold, use_half, img_size, num_images_to_process, device, progress_value, lock) for i in range(config["num_processes"])]

            # Run the model_worker function in parallel
            results = pool.starmap_async(model_worker, args)

            # Update progress bar
            while not results.ready():
                progress_bar.progress(progress_value.value / num_images_to_process)
                time.sleep(0.1)

            results = results.get()

            # Collect results from all processes
            for result in results:
                local_mislabeled_images, local_tp, local_fp, local_fn = result
                mislabeled_images.extend(local_mislabeled_images)
                tp += local_tp
                fp += local_fp
                fn += local_fn

            print(f"Total TP: {tp}, FP: {fp}, FN: {fn}")

        progress_bar.empty()

# Initialize the default model loaded flag
default_model_loaded = False

# Select model
uploaded_model = st.file_uploader("Upload your YOLOv8 model file", type=["pt"], key="model_uploader", accept_multiple_files=False)
model_paths = []
default_model_path = Path(config["default_model_path"])
if uploaded_model:
    uploaded_model_content = uploaded_model.read()
    for i in range(config["num_processes"]):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model_file:
            tmp_model_file.write(uploaded_model_content)
            model_paths.append(tmp_model_file.name)
else:
    for i in range(config["num_processes"]):
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

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Add a button to start processing images
if st.button("Process Images"):
    if model_paths and dataset_dir:
        dataset_splits = config["dataset_splits"]
        mislabeled_images = []

        total_images = 0
        for split in dataset_splits:
            total_images += len(load_images(dataset_dir / split / 'images'))

        num_images_to_process = int(total_images * (percentage / 100))

        all_images = []
        for split in dataset_splits:
            all_images.extend(load_images(dataset_dir / split / 'images'))

        if num_images_to_process < total_images:
            all_images = np.random.choice(all_images, num_images_to_process, replace=False).tolist()

        run_processing(model_paths, all_images, initial_iou_threshold, conf_threshold, use_half, img_size, num_images_to_process, device)

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
                image_with_annotations = draw_polygons(img_path, result, labels, suspect_boxes, class_names)
            else:
                image_with_annotations = draw_boxes(img_path, result, label_boxes, suspect_boxes, class_names)

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
