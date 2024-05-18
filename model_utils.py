import numpy as np
import cv2
import os
from image_utils import get_bounding_box, calculate_iou
import gc

def categorize_predictions(predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.5, image_width=4096, image_height=3072):
    tp, fp, fn = 0, 0, 0
    matched_gts = []

    for pred in predictions:
        pred_box, pred_conf, pred_class = pred['box'], pred['confidence'], pred['class']
        if pred_conf < conf_threshold:
            continue
        
        matched = False
        for gt in ground_truths:
            gt_box, gt_class = gt['box'], gt['class']
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_class == gt_class:
                tp += 1
                matched_gts.append(gt)
                matched = True
                break

        if not matched:
            fp += 1

    for gt in ground_truths:
        if gt not in matched_gts:
            fn += 1

    return tp, fp, fn

def process_image(model, image_path, iou_threshold=0.5, conf=0.2, half=True, imgsz=640, class_names=None):
    try:
        results = model(str(image_path), half=half, conf=conf, imgsz=imgsz)
        result = results[0]
        label_path = str(image_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        if not os.path.exists(label_path):
            return (image_path, result, ['NULL'], [], [], 0, 1, 0)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
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

        prediction_bounding_boxes = [{'box': [x1, y1, x2, y2], 'confidence': box.conf[0], 'class': int(box.cls[0])} for box in result.boxes for x1, y1, x2, y2 in [map(int, box.xyxy[0])]]

        tp, fp, fn = categorize_predictions(prediction_bounding_boxes, label_bounding_boxes, iou_threshold, conf, width, height)

        # Release memory
        del image
        gc.collect()

        return (image_path, result, labels, label_bounding_boxes, prediction_bounding_boxes, tp, fp, fn)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
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
                    if iou >= threshold and class_id == label_box['class']:
                        match_found = True
                        break
                
                if not match_found:
                    new_suspect_boxes.append({'box': pred_box, 'class': class_id})
            
            if new_suspect_boxes:
                filtered_images.append((img_path, result, labels, label_boxes, new_suspect_boxes, tp, fp, fn))
    return filtered_images
