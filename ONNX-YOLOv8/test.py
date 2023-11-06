import requests

url = 'http://0.0.0.0:8000/yolov8_onnx_inference'
file = {'file': open('/home/duong0411/PycharmProjects/yolo_onnx/data_test/data_v4_50.jpg', 'rb')}
resp = requests.post(url=url, files=file)
print(resp.json())