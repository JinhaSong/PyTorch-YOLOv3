from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import zipfile


class YOLOv3:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.config_path = os.path.join(os.getcwd(), "config", "yolov3-nsfw.cfg")
        self.weights_path = os.path.join(os.getcwd(), "weights", "yolov3-nsfw_50000.weights")
        self.class_path = os.path.join(os.getcwd(), "data", "nsfw.names")
        self.zip_path = "D:\\Projects\\GitRepo\\VideoPreviewer\\yolov3\\test.zip"
        self.img_size = 416
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.batch_size = 1
        self.n_cpu = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Darknet(self.config_path, img_size=self.img_size).to(device)

        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))

        self.model.eval()  # Set in evaluation mode

        self.classes = load_classes(self.class_path)  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def inference_by_path(self, zip_path):
        dataloader = DataLoader(
            # TarLoader("D:\\Projects\\GitRepo\\VideoPreviewer\\yolov3\\test.tar", img_size=img_size),
            ZipFileLoader(zip_path, img_size=self.img_size),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
        )

        imgs = []
        img_detections = []

        prev_time = time.time()
        for batch_i, (input_name, input_imgs) in enumerate(dataloader):
            input_imgs = Variable(input_imgs.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

            imgs.append(input_name)
            img_detections.extend(detections)

        zip_file = zipfile.ZipFile(zip_path)

        results = []
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            img_name = path[0]
            img = zip_file.read(img_name)
            img = np.array(Image.open(io.BytesIO(img)))

            img_result = {
                "image_name": img_name,
                "result": []
            }
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.img_size, img.shape[:2])
                for x, y, x2, y2, conf, cls_conf, cls_pred in detections:
                    w = x2 - x
                    h = y2 - y
                    img_result['result'].append({
                        "position": {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                        },
                        "description": [{
                            "description": self.classes[int(cls_pred)],
                            "score": cls_conf.item()
                        }]
                    })
            results.append(img_result)

        zip_file.close()
        self.result = results

        return self.result


yolov3 = YOLOv3()
print(yolov3.inference_by_path("D:\\Projects\\GitRepo\\VideoPreviewer\\yolov3\\test.zip"))