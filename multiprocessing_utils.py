# multiprocessing_utils.py
from ultralytics import YOLO
import torch
import asyncio
from model_utils import process_image

async def process_image_async(model, image_path, iou_threshold, conf_threshold, use_half, img_size):
    return await process_image(model, image_path, iou_threshold, conf_threshold, use_half, img_size)

def model_worker(model_path, images, device, results_queue, iou_threshold, conf_threshold, use_half, img_size, total_images):
    local_mislabeled_images = []
    tp, fp, fn = 0, 0, 0

    model = YOLO(model_path).to(device)
    
    async def process_images():
        nonlocal tp, fp, fn
        for image_path in images:
            result = await process_image_async(model, image_path, iou_threshold, conf_threshold, use_half, img_size)
            if result:
                image_path, _, _, _, _, local_tp, local_fp, local_fn = result
                tp += local_tp
                fp += local_fp
                fn += local_fn
                local_mislabeled_images.append(result)
            results_queue.put(1)
    
    asyncio.run(process_images())

    results_queue.put((local_mislabeled_images, tp, fp, fn))
