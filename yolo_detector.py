"""
Wrapper for YOLO v5 detector from Ultralytics

COGNITECHNA spol. s r.o.
Roman Juranek <r.juranek@cognitechna.cz>

---

List of available classes can be found here:
https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml


"""

import cv2
import numpy as np
import torch
from shapely.geometry import box
from typing import Iterable

from detection import ObjectObservation


class YOLODetector:
    default_classes = [
        "person", "bicycle", "car", "motorcycle", "bus", "truck"
    ]

    def __init__(self, model: str = "yolov5l6", classes:Iterable[str]=None, max_size: int = 640, min_score:float = 0.3):
        self.model = torch.hub.load("ultralytics/yolov5", model, pretrained=True)
        self.model.agnostic = False
        self.model.iou = 0.7
        classes = classes or YOLODetector.default_classes
        class_names = self.model.names
        self.model.classes = list(map(class_names.index, filter(lambda name: name in class_names, classes)))
        self.model.conf = min_score
        self.max_size = max_size

    @staticmethod
    def from_dict(d:dict) -> "YOLODetector":
        return YOLODetector(
            model=d.get("model", "yolov5n6"),
            classes=d.get("classes"),
            max_size=d.get("max_size", 1024),
            min_score=d.get("min_score", 0.3),
        )

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
