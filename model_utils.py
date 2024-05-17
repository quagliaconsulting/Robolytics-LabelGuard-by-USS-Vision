import numpy as np
import cv2
import os
from image_utils import get_bounding_box

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area != 0 else 0

    print(f"Box1: {box1}, Box2: {box2}, IoU: {iou}")
    
    return iou

def categorize_predictions(predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.5, image_width=4096, image_height=3072):
    tp = 0
    fp = 0
    fn = 0

    matched_gts = []

    for pred in predictions:
        pred_box, pred_conf, pred_class = pred['box'], pred['confidence'], pred['class']
        if pred_conf < conf_threshold:
            continue
        
        # Normalize prediction box coordinates
        pred_box = [
            pred_box[0] / image_width,
            pred_box[1] / image_height,
            pred_box[2] / image_width,
            pred_box[3] / image_height
        ]
        
        matched = False
        for gt in ground_truths:
            gt_box, gt_class = gt['box'], gt['class']
            iou = calculate_iou(pred_box, gt_box)
            print(f"Prediction: {pred_box}, Ground Truth: {gt_box}, IoU: {iou}")
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

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    
    return tp, fp, fn

async def process_image(model, image_path, iou_threshold=0.5, conf=0.2, half=True, imgsz=640):
    try:
        results = model(str(image_path), half=half, conf=conf, imgsz=imgsz)
        result = results[0]
        label_path = str(image_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        # Handle NULL images
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
