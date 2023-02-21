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
import flask_socketio
from queue import Queue

import detection
from detection import ObjectObservation

from era_5g_object_detection_common.image_detector import ImageDetector, ImageDetectorInitializationFailed
from era_5g_object_detection_standalone.worker import Worker


class YOLODetector(Worker, ImageDetector):
    default_classes = [
        "person", "bicycle", "car", "motorcycle", "bus", "truck"
    ]

    def __init__(self,
                 image_queue: Queue,
                 app,
                 model: str = "yolov5l6",
                 classes: Iterable[str] = None,
                 max_size: int = 640,
                 min_score: float = 0.3,
                 filter_in_frame: bool = True,
                 min_area: float = None,
                 **kw
                 ):
        super().__init__(image_queue=image_queue, app=app, **kw)
        self.model = torch.hub.load("ultralytics/yolov5", model, pretrained=True)
        self.model.agnostic = False
        self.model.iou = 0.7
        classes = classes or YOLODetector.default_classes
        if classes is not None:
            # Init detected classes 
            # Inverted name index: name -> class_id
            name_idx = dict(((name, class_id) for class_id, name in self.model.names.items()))
            # List class_id specified by names in classes passed as parameter, ignoring unknown classes
            self.model.classes = [name_idx[nm] for nm in classes if
                                  nm in name_idx] or None  # ... or None - in case og empty list leave None value not
            # empty list
        self.model.conf = min_score
        self.max_size = max_size
        self.filter_in_frame = filter_in_frame
        self.min_area = min_area

    @staticmethod
    def from_dict(d: dict) -> "YOLODetector":
        return YOLODetector(
                image_queue=None,
                app=None,
                model=d.get("model", "yolov5n6"),
                classes=d.get("classes"),
                max_size=d.get("max_size", 1024),
                min_score=d.get("min_score", 0.3),
                filter_in_frame=d.get("filter_in_frame", False),
                min_area=d.get("min_area"),
        )

    def process_image(self, image):
        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = max(h, w) / self.max_size  # shape (1080, 1920), max_size=960 -> scale=1920/960 = 2
            dst_size = (int(w // scale), int(h // scale))
            image = cv2.resize(image, dst_size, interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1

        # Run detection
        res = self.model(np.transpose(image, [2, 0, 1]))

        # Convert detections
        det = res.xyxy[0].cpu().numpy()
        rects, scores, labels = np.split(det, [4, 5], axis=1)
        labels = labels.ravel().astype(np.int32).tolist()
        scores = scores.ravel().tolist()

        # Convert coords to shapely Polygon
        geometries = map(lambda x: box(*x), rects * scale)

        # Generator of object instances
        all_dets = (
            ObjectObservation(
                    geometry=geometry,
                    score=score,
                    label=label,
            )
            for geometry, label, score in zip(geometries, labels, scores)
        )

        # Filter objects that are in the frame
        if self.filter_in_frame:
            is_in_frame = lambda d: d.is_in_frame((h, w), margin=10)
            all_dets = filter(is_in_frame, all_dets)

        # Filter objects with sufficient size
        if self.min_area is not None and self.min_area > 0:
            sufficient_size = lambda d: d.geometry.area > self.min_area
            all_dets = filter(sufficient_size, all_dets)

        return list(all_dets)

    def publish_results(self, results, metadata):
        """
        Publishes the results to the robot

        Args:
            metadata (_type_): NetApp-specific metadata related to processed image.
            results (_type_): The results of the detection.
        """
        detections = list()

        if results is not None:
            for result in results:
                det = dict()
                det["bbox"] = result.bounds()
                det["score"] = result.score
                det["class"] = result.label
                det["class_name"] = self.model.names[result.label]

                detections.append(det)

            r = {"timestamp": metadata["timestamp"],
                 "detections": detections}

            # use the flask app to return the results
            with self.app.app_context():
                # print(f"publish_results to: {metadata['websocket_id']} flask_socketio.send: {r}")
                flask_socketio.send(r, namespace='/results', to=metadata["websocket_id"])
