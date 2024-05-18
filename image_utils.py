import cv2
import numpy as np
from pathlib import Path

def load_images(directory: Path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [str(path) for path in directory.rglob('*') if path.suffix.lower() in image_extensions]
    return images

def draw_polygons(image_path, result, labels, suspect_boxes, class_names):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    height, width, _ = image.shape

    for label in labels:
        try:
            parts = label.strip().split()
            class_id = int(parts[0])
            points = list(map(float, parts[1:]))
            polygon = np.array(points).reshape((-1, 2)) * [width, height]
            polygon = polygon.astype(int)
            cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, class_names[class_id] if class_id < len(class_names) else f"Class {class_id}", (polygon[0][0], polygon[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except ValueError:
            continue

    # Draw suspect boxes with class names
    for box in suspect_boxes:
        try:
            x1, y1, x2, y2 = box['box']
            class_id = box['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except ValueError:
            continue

    return image

def draw_boxes(image_path, result, label_boxes, suspect_boxes, class_names):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    height, width, _ = image.shape

    # Draw label boxes with class names
    for label_box in label_boxes:
        try:
            x1, y1, x2, y2 = label_box['box']
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            class_id = label_box['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except ValueError:
            continue

    # Draw suspect boxes with class names
    for box in suspect_boxes:
        try:
            x1, y1, x2, y2 = box['box']
            class_id = box['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except ValueError:
            continue

    return image

def get_bounding_box(polygon, img_width, img_height):
    x_coords = polygon[:, 0] * img_width
    y_coords = polygon[:, 1] * img_height
    x_min = int(np.min(x_coords))
    y_min = int(np.min(y_coords))
    x_max = int(np.max(x_coords))
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