import flask_socketio
from queue import Queue

from yolo_detector import YOLODetector

from era_5g_object_detection_common.image_detector import ImageDetector
from era_5g_object_detection_standalone.worker import Worker


class YOLODetectorWorker(Worker, ImageDetector):
    def __init__(self,
                 image_queue: Queue,
                 app,
                 config: dict,
                 **kw
                 ):
        super().__init__(image_queue=image_queue, app=app, **kw)
        self.detector = YOLODetector.from_dict(config)

    def process_image(self, image):
        return self.detector.detect(image)

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
                det["class_name"] = self.detector.model.names[result.label]

                detections.append(det)

            r = {"timestamp": metadata["timestamp"],
                 "detections": detections}

            # use the flask app to return the results
            with self.app.app_context():
                # print(f"publish_results to: {metadata['websocket_id']} flask_socketio.send: {r}")
                flask_socketio.send(r, namespace='/results', to=metadata["websocket_id"])
