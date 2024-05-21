# processing.py
import sys
import yaml
from ultralytics import YOLO
from image_utils import load_images, draw_boxes
from model_utils import process_image, filter_mislabeled_images
import torch
import logging
from multiprocessing import Pool
import gc

# Configure logging
logging.basicConfig(filename='processing.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def model_worker(args):
    model_path, images, iou_threshold, conf_threshold, use_half, img_size, device, class_names, progress_counters, index = args
    model = YOLO(model_path).to(device)
    local_mislabeled_images = []
    tp, fp, fn = 0, 0, 0

    for image_path in images:
        result = process_image(model, image_path, iou_threshold, conf_threshold, bool(use_half), img_size, class_names)
        if result:
            image_path, _, _, _, _, local_tp, local_fp, local_fn = result
            tp += local_tp
            fp += local_fp
            fn += local_fn
            local_mislabeled_images.append(result)
        
        # Update the progress counter
        progress_counters[index] += 1

    # Free up memory
    # #del model
    # #torch.cuda.empty_cache()
    # gc.collect()

    return local_mislabeled_images, tp, fp, fn

if __name__ == "__main__":
    try:
        args_file = sys.argv[1]
        results_file = sys.argv[2]

        with open(args_file, 'r') as file:
            args = yaml.safe_load(file)

        model_paths = args["model_paths"]
        all_images = args["all_images"]
        initial_iou_threshold = args["initial_iou_threshold"]
        conf_threshold = args["conf_threshold"]
        use_half = args["use_half"]
        img_size = args["img_size"]
        num_images_to_process = args["num_images_to_process"]
        device = args["device"]
        class_names = args["class_names"]
        output_dir = args.get("output_dir", "output")

        images_per_model = len(all_images) // len(model_paths)
        tasks = []

        for i, model_path in enumerate(model_paths):
            images = all_images[i * images_per_model: (i + 1) * images_per_model]
            tasks.append((model_path, images, initial_iou_threshold, conf_threshold, use_half, img_size, device, class_names))

        # Use multiprocessing to process images in parallel
        with Pool(processes=len(model_paths)) as pool:
            logging.info("Starting multiprocessing pool")
            results = pool.map(model_worker, tasks)
            logging.info("Finished multiprocessing pool")

        mislabeled_images = []
        tp, fp, fn = 0, 0, 0

        logging.info("Starting to aggregate results")
        for result in results:
            local_mislabeled_images, local_tp, local_fp, local_fn = result
            mislabeled_images.extend(local_mislabeled_images)
            tp += local_tp
            fp += local_fp
            fn += local_fn
        logging.info("Finished aggregating results")

        output = {
            "mislabeled_images": mislabeled_images,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

        logging.info("Writing results to file")
        with open(results_file, 'w') as file:
            yaml.dump(output, file)
        logging.info("Finished writing results to file")

        logging.info("Processing completed successfully.")
        print("done")
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        print(f"Error during processing: {e}")
