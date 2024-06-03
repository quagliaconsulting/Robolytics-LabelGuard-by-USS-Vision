import streamlit as st
import yaml
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import streamlit_image_zoom
from image_utils import draw_polygons, draw_boxes, load_images
from model_utils import process_image, filter_mislabeled_images
from ui_components import add_logos, main_ui
import torch
import logging
from multiprocessing import Pool, Manager
from ultralytics import YOLO
from processing import model_worker
import threading
import gc
import time

# Configure logging
logging.basicConfig(filename='processing.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

def main():
    # Load configuration
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Initialize Streamlit UI and load configuration
    add_logos()
    st.text("")
    main_ui()

    # Load class names from data.yaml
    dataset_dir = Path(config["dataset_directory"])
    with open(dataset_dir / "data.yaml", "r") as data_file:
        data_config = yaml.safe_load(data_file)
    class_names = data_config["names"]

    # Initialize the default model loaded flag
    default_model_loaded = False

    # Create a shared manager
    manager = Manager()
    progress_counters = manager.list([0] * config["num_processes"])

    # Select model
    uploaded_model = st.file_uploader("Upload your YOLOv8 model file", type=["pt"], key="model_uploader", accept_multiple_files=False)
    model_paths = []
    default_model_path = Path(config["default_model_path"])
    if uploaded_model:
        uploaded_model_content = uploaded_model.read()
        for i in range(config["num_processes"]):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model_file:
                tmp_model_file.write(uploaded_model_content)
                model_paths.append(str(tmp_model_file.name))
    else:
        for i in range(config["num_processes"]):
            model_paths.append(str(default_model_path))
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

            total_images = 0
            for split in dataset_splits:
                total_images += len(load_images(dataset_dir / split / 'images'))

            num_images_to_process = int(total_images * (percentage / 100))

            all_images = []
            for split in dataset_splits:
                all_images.extend(load_images(dataset_dir / split / 'images'))

            if num_images_to_process < total_images:
                all_images = np.random.choice(all_images, num_images_to_process, replace=False).tolist()
                
            tasks = []
            images_per_model = len(all_images) // len(model_paths)

            for i, model_path in enumerate(model_paths):
                images = all_images[i * images_per_model: (i + 1) * images_per_model]
                tasks.append((model_path, images, initial_iou_threshold, conf_threshold, use_half, img_size, device, class_names, progress_counters, i))


            # Display the progress bar
            progress_bar = st.progress(0)

            # Use multiprocessing to process images in parallel
            with Pool(processes=len(model_paths)) as pool:
                results = pool.map_async(model_worker, tasks)

                while not results.ready():
                    progress_bar.progress(sum(progress_counters) / num_images_to_process)
                    time.sleep(0.1)

                pool.close()
                pool.join()

            # Aggregate results
            mislabeled_images = []
            tp, fp, fn = 0, 0, 0

            for result in results.get():
                local_mislabeled_images, local_tp, local_fp, local_fn = result
                mislabeled_images.extend(local_mislabeled_images)
                tp += local_tp
                fp += local_fp
                fn += local_fn

            # Clear large intermediate results from memory
            del all_images
            gc.collect()

            # Save results in session state
            st.session_state["mislabeled_images"] = mislabeled_images
            st.session_state["tp"] = tp
            st.session_state["fp"] = fp
            st.session_state["fn"] = fn

    # Add debug statements to the end of the image processing section
    if "mislabeled_images" in st.session_state:
        display_mode = st.radio("Display Mode", ["Polygons", "Bounding Boxes"])

        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, initial_iou_threshold, step=0.01, key="post_inference_threshold")

        mislabeled_images = st.session_state["mislabeled_images"]
        tp = st.session_state["tp"]
        fp = st.session_state["fp"]
        fn = st.session_state["fn"]

        st.write(f"Total TP: {tp}, FP: {fp}, FN: {fn}")

        filtered_mislabeled_images = filter_mislabeled_images(mislabeled_images, iou_threshold)

        logging.debug(f"Filtered Mislabeled Images: {filtered_mislabeled_images}")

        if filtered_mislabeled_images:
            st.write(f"Found {len(filtered_mislabeled_images)} potentially mislabeled images:")
            for img_path, result, labels, label_boxes, suspect_boxes, tp, fp, fn in filtered_mislabeled_images:
                st.write(f"Image: {img_path}")
                st.write(f"TP: {tp}, FP: {fp}, FN: {fn}")

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

                # Clear memory after displaying each image
                del image_with_annotations
                gc.collect()
        else:
            st.write("No potentially mislabeled images found.")
    else:
        st.write("Please upload a model file and enter the dataset directory to start processing images.")



if __name__ == '__main__':
    main()
