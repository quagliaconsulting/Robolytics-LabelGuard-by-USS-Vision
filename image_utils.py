import cv2
import numpy as np
from pathlib import Path
import logging

def load_images(directory: Path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [str(path) for path in directory.rglob('*') if path.suffix.lower() in image_extensions]
    return images

def draw_polygons(image_path, result, labels, prediction_boxes, class_names):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Failed to read image {image_path}")
        return None

    height, width, _ = image.shape

    # Draw all label boxes based on their status
    for label in labels:
        try:
            parts = list(map(float, label.strip().split()))
            if len(parts) % 2 == 1:
                class_id = int(parts[0])
                points = np.array(parts[1:]).reshape(-1, 2)
                points[:, 0] *= width
                points[:, 1] *= height
                points = points.astype(int)
                # Default status if not specified
                status = 'FN' if 'FN' in label else 'TP'
                color = (0, 0, 255) if status == 'FN' else (255, 0, 0)  # Red for FN, Blue for TP
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                cv2.putText(image, class_names[class_id] if class_id < len(class_names) else f"Class {class_id}", 
                            (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                logging.error(f"Invalid label format for polygon: {label}")
                continue
        except Exception as e:
            logging.error(f"Error drawing polygon for label: {label}. Error: {e}")
            continue

    # Draw all prediction boxes based on their status
    for box in prediction_boxes:
        try:
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            class_id = box['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            if box.get('status') == 'TP':
                color = (255, 0, 0)  # Blue for TP
            elif box.get('status') == 'FP':
                color = (0, 165, 255)  # Bright orange for FP
            else:
                color = (0, 0, 255)  # Red for FN

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            logging.error(f"Error drawing rectangle for prediction: {box}. Error: {e}")
            continue

    return image




def draw_boxes(image_path, result, label_boxes, prediction_boxes, class_names):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Failed to read image {image_path}")
        return None

    height, width, _ = image.shape

    # Draw all label boxes based on their status
    for label_box in label_boxes:
        try:
            x1, y1, x2, y2 = label_box['box']
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            class_id = label_box['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            color = (0, 0, 255) if label_box.get('status', 'FN') == 'FN' else (255, 0, 0)  # Red for FN, Blue for TP
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except ValueError as e:
            logging.error(f"Error drawing label box: {label_box}. Error: {e}")
            continue

    # Draw all prediction boxes based on their status
    for pred_box in prediction_boxes:
        try:
            x1, y1, x2, y2 = pred_box['box']
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            class_id = pred_box['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            if pred_box.get('status') == 'TP':
                color = (255, 0, 0)  # Blue for TP
            elif pred_box.get('status') == 'FP':
                color = (0, 165, 255)  # Bright orange for FP
            else:
                color = (0, 0, 255)  # Red for FN

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except ValueError as e:
            logging.error(f"Error drawing prediction box: {pred_box}. Error: {e}")
            continue

    return image



def get_bounding_box(polygon, img_width, img_height):
    x_coords = polygon[:, 0] * img_width
    y_coords = polygon[:, 1] * img_height
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))
    return [x_min / img_width, y_min / img_height, x_max / img_width, y_max / img_height]


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

    return intersection_area / union_area
