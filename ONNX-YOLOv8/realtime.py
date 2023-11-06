import cv2
from yolov8 import YOLOv8
import yaml

cap = cv2.VideoCapture(0)
config_fp = "/home/duong0411/PycharmProjects/yolo_onnx/ONNX-YOLOv8/configs/yolov8_onnx.yaml"
with open(config_fp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
yolov8_onnx = YOLOv8(config)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    preds = yolov8_onnx.inference(frame)
    for result in preds:
        for box in result:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 1)
    cv2.imshow("Detected Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
