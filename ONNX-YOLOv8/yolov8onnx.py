import cv2
import yaml
from yolov8 import YOLOv8
import os
import time
import numpy as np
if __name__ == "__main__":
    config_fp = "/home/duong0411/PycharmProjects/yolo_onnx/ONNX-YOLOv8/configs/yolov8_onnx.yaml"
    input_dir = "/home/duong0411/PycharmProjects/yolo_onnx/data_test/data_v4_50.jpg"
    output_dir = "/home/duong0411/PycharmProjects/yolo_onnx/result1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

    with open(config_fp, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    total_time = 0

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        imgs = cv2.imread(image_path)

        yolov8_onnx = YOLOv8(config)
        start = time.perf_counter()

        preds = yolov8_onnx.inference(imgs)
        elapsed_time = time.perf_counter() - start
        total_time += elapsed_time

        for i, det in enumerate(preds):
            for j,box in enumerate(det):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Crop the detected object from the image
                cropped_image = imgs[y1:y2, x1:x2]
                output_path = os.path.join(output_dir, f"object_{i}_{j}_{image_file}")
                # Draw a rectangle around the detected object
                cv2.rectangle(imgs, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.imwrite(output_path,cropped_image)


    print(f"Total inference time for {len(image_files)} images: {total_time:.2f} seconds")
