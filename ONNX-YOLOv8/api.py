import cv2
import yaml
from yolov8 import YOLOv8
from fastapi import FastAPI, File, UploadFile
import numpy as np
from def_recognition_10_10 import Recognition
config_fp = "/home/duong0411/PycharmProjects/yolo_onnx/ONNX-YOLOv8/configs/yolov8_onnx.yaml"
with open(config_fp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

yolov8_onnx = YOLOv8(config)
rec_model_dir = "/home/duong0411/PycharmProjects/yolo_onnx/ONNX-YOLOv8/models/rec_10_2.onnx"
rec_char_dict_path = "/home/duong0411/PycharmProjects/yolo_onnx/ONNX-YOLOv8/models/dict_9_29.txt"
app = FastAPI()
@app.post("/yolov8_onnx_inference1")
async def yolov8_onnx_inference(file: UploadFile):
    content = await file.read()
    image_buffer = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    preds = yolov8_onnx.inference(image)
    results = []
    for i, det in enumerate(preds):
        detected_objects = []
        for box in det:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_image = image[y1:y2, x1:x2]
            detected_objects.append(cropped_image)
        recognition_results = Recognition(detected_objects, rec_model_dir, rec_char_dict_path)
        objects = []
        for j, box in enumerate(det):
            obj = {}
            obj["text"] = recognition_results[j]  # Add the recognized text to the object
            objects.append(obj)
        results.append(objects)

    return results
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)