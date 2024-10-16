# import torch
# import cv2
# from PIL import Image
# import numpy as np
# from fastapi import UploadFile
# # Load YOLOv5 model for object detection
# yolo_model = torch.hub.load(
#     'ultralytics/yolov5', 'custom', path='models/yolov5_model.pt')


# def count_items(image_visible: UploadFile, image_ir: UploadFile) -> int:
#     # Read and preprocess visible and IR images
#     visible_img = Image.open(image_visible.file)
#     ir_img = Image.open(image_ir.file)

#     # Convert images to OpenCV format
#     visible_img_cv = np.array(visible_img)
#     ir_img_cv = np.array(ir_img)

#     # YOLOv5 inference on visible image
#     results_visible = yolo_model(visible_img_cv)
#     count_visible = len(results_visible.xyxy[0])

#     # YOLOv5 inference on IR image
#     results_ir = yolo_model(ir_img_cv)
#     count_ir = len(results_ir.xyxy[0])

#     # Confirm count by averaging counts from visible and IR images
#     final_count = (count_visible + count_ir) // 2

#     return final_count
