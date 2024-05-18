import numpy as np
import cv2
import os
from image_utils import get_bounding_box, calculate_iou
import logging

def categorize_predictions(predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.5, image_width=4096, image_height=3072):
    tp, fp, fn = 0, 0, 0
    matched_gts = []

    for pred in predictions:
        pred_box, pred_conf, pred_class = pred['box'], pred['confidence'], pred['class']
        logging.debug(f"Processing prediction: {pred_box}, confidence: {pred_conf}, class: {pred_class}")
        if pred_conf < conf_threshold:
            logging.debug(f"Prediction confidence {pred_conf} is below threshold {conf_threshold}")
            continue
        
        matched = False
        for gt in ground_truths:
            gt_box, gt_class = gt['box'], gt['class']
            iou = calculate_iou(pred_box, gt_box)
            logging.debug(f"Comparing with ground truth: {gt_box}, class: {gt_class}, IoU: {iou}")
            if iou >= iou_threshold and pred_class == gt_class:
                tp += 1
                matched_gts.append(gt)
                matched = True
                logging.debug(f"Match found. IoU: {iou}, class: {pred_class}. Incrementing TP")
                break

        if not matched:
            fp += 1
            logging.debug(f"No match found. Incrementing FP")

    for gt in ground_truths:
        if gt not in matched_gts:
            fn += 1
            logging.debug(f"Ground truth {gt} not matched. Incrementing FN")

    logging.info(f"Categorized predictions. TP: {tp}, FP: {fp}, FN: {fn}")
    return tp, fp, fn

def process_image(model, image_path, iou_threshold=0.5, conf=0.2, half=True, imgsz=None, class_names=None):
    try:
        logging.info(f"Processing image: {image_path}")
        results = model(str(image_path), half=half, conf=conf, imgsz=imgsz)
        result = results[0]
        label_path = str(image_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        # Handle NULL images
        if not os.path.exists(label_path):
            logging.warning(f"Label file not found for image {image_path}. Assuming NULL labels")
            return (image_path, result, ['NULL'], [], [], 0, 1, 0)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logging.error(f"Failed to read image {image_path}")
            return None
        height, width, _ = image.shape

        label_bounding_boxes = []
        for label in labels:
            parts = list(map(float, label.strip().split()))
            class_id = int(parts[0])
            points = parts[1:]
            if len(points) % 2 == 0:
                polygon = np.array(points).reshape((-1, 2))
                bounding_box = get_bounding_box(polygon, width, height)
                label_bounding_boxes.append({'box': bounding_box, 'class': class_id})
                logging.debug(f"Added label bounding box: {bounding_box}, class: {class_id}")

        prediction_bounding_boxes = [{'box': [x1 / width, y1 / height, x2 / width, y2 / height], 'confidence': box.conf[0], 'class': int(box.cls[0])} for box in result.boxes for x1, y1, x2, y2 in [map(int, box.xyxy[0])]]
        logging.debug(f"Prediction bounding boxes: {prediction_bounding_boxes}")

        tp, fp, fn = categorize_predictions(prediction_bounding_boxes, label_bounding_boxes, iou_threshold, conf, width, height)
        logging.info(f"Processed image: {image_path}, TP: {tp}, FP: {fp}, FN: {fn}")
        return (image_path, result, labels, label_bounding_boxes, prediction_bounding_boxes, tp, fp, fn)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
    return None

def filter_mislabeled_images(mislabeled_images, threshold):
    filtered_images = []
    for img_path, result, labels, label_boxes, prediction_bounding_boxes, tp, fp, fn in mislabeled_images:
        if fp > 0 or fn > 0:
            new_suspect_boxes = []
            for box in prediction_bounding_boxes:
                x1, y1, x2, y2 = box['box']
                pred_box = [x1, y1, x2, y2]
                class_id = box['class']

                match_found = False
                for label_box in label_boxes:
                    iou = calculate_iou(pred_box, label_box['box'])
                    logging.debug(f"Comparing pred_box: {pred_box} with label_box: {label_box['box']}, IoU: {iou}")
                    if iou >= threshold and class_id == label_box['class']:
                        match_found = True
                        break
                
                if not match_found:
                    new_suspect_boxes.append({'box': pred_box, 'class': class_id})
                    logging.debug(f"Added suspect box: {pred_box}, class: {class_id}")
            
            if new_suspect_boxes:
                filtered_images.append((img_path, result, labels, label_boxes, new_suspect_boxes, tp, fp, fn))
                logging.info(f"Filtered image: {img_path}, new suspect boxes: {new_suspect_boxes}")

    return filtered_images
