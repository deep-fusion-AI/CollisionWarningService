"""
Wrapper for YOLO v5 detector from Ultralytics

COGNITECHNA spol. s r.o.
Roman Juranek <r.juranek@cognitechna.cz>

---

TODO
* Configure class names in constructor - list of names

"""

import cv2
import numpy as np
import torch
from shapely.geometry import box

from detection import ObjectObservation

_traffic_classes = {0, 1, 2, 3, 5, 7, 16}


class YOLODetector:
    def __init__(self, model: str = "yolov5l6", max_size: int = 640):
        self.model = torch.hub.load("ultralytics/yolov5", model, pretrained=True)
        self.model.agnostic = True  # NMS will be done as class-agnostic
        self.model.classes = list(_traffic_classes)
        self.model.conf = 0.6
        self.max_size = max_size

    def detect(self, image):
        h,w = image.shape[:2]
        if max(h,w) > self.max_size:
            scale = max(h,w) / self.max_size  # shape (1080, 1920), max_size=960 -> scale=1920/960 = 2
            dst_size = (int(w // scale), int(h // scale))
            image = cv2.resize(image, dst_size, interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1

        # Run detection
        res = self.model(np.transpose(image, [2, 0, 1]))

        # Convert detections
        det = res.xyxy[0].cpu().numpy()
        rects, scores, labels = np.split(det, [4, 5], axis=1)
        labels = labels.ravel().astype(np.int).tolist()
        scores = scores.ravel().tolist()

        geometries = list(map(lambda x: box(*x), rects * scale))
        
        return [
            ObjectObservation( 
                geometry=geometry,
                score=score,
                label=label,
            )
            for geometry, label, score in zip(geometries, labels, scores)
        ]
