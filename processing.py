import sys
import yaml
from ultralytics import YOLO
from image_utils import load_images, draw_polygons, draw_boxes, get_bounding_box
from model_utils import process_image, filter_mislabeled_images
import torch

def model_worker(args):
    model_path, images, iou_threshold, conf_threshold, use_half, img_size, total_images, device, class_names = args
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

    return local_mislabeled_images, tp, fp, fn

if __name__ == "__main__":
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

    images_per_model = len(all_images) // len(model_paths)
    tasks = []
    tp, fp, fn = 0, 0, 0

    results = []
    for i, model_path in enumerate(model_paths):
        images = all_images[i * images_per_model: (i + 1) * images_per_model]
        result = model_worker((model_path, images, initial_iou_threshold, conf_threshold, use_half, img_size, num_images_to_process, device, class_names))
        results.append(result)

    mislabeled_images = []
    for result in results:
        local_mislabeled_images, local_tp, local_fp, local_fn = result
        mislabeled_images.extend(local_mislabeled_images)
        tp += local_tp
        fp += local_fp
        fn += local_fn

    output = {
        "mislabeled_images": mislabeled_images,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

    with open(results_file, 'w') as file:
        yaml.dump(output, file)

    print("done")
