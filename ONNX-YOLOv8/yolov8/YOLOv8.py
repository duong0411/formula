import time
import cv2
import numpy as np
import onnxruntime
from yolov8.utils import letterbox,scale_boxes,non_max_suppression
import yaml
import copy
import torch
class YOLOv8:
    def __init__(self, config: yaml) -> None:
        self.config = config
        self.onnx = self.config["onnx"]
        self.size = self.config["size"]
        self.batch_szie = self.config["batch_szie"]
        self.stride = self.config["stride"]
        self.device = self.config["device"]
        self.confidence_threshold = self.config["confidence_threshold"]
        self.iou_threshold = self.config["iou_threshold"]

        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self.device == "gpu":
            providers = ["CUDAExecutionProvider"]
        else:
            # TODO: logging
            print(f"Does not support {self.device}")

        self.sess = onnxruntime.InferenceSession(self.onnx, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def pre_process(self, images: list):
        imgs = []
        for img in images:
            img = letterbox(img, self.size, stride=self.stride, auto=False)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = img / 255
            imgs.append(img)
        imgs = np.array(imgs).astype(np.float32)
        return imgs

    def post_process(self, pred: np.array, images, ori_images):
        pred = [torch.from_numpy(pred)]
        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold,None, False, 3000)

        post_det = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(images.shape[2:], det[:, :4], ori_images[i].shape).round()
                det = det.detach().cpu().numpy()
                post_det.append(det)

        return post_det

    def inference(self, images: list):
        if not isinstance(images, list):
            images = [images]
        ori_images = copy.deepcopy(images)
        images = self.pre_process(images)

        pred = self.sess.run(None, {self.input_name: images})[0]
        pred = self.post_process(pred, images, ori_images)

        return pred



