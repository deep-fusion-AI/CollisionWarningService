from multiprocessing import Queue
from threading import Thread, Event
from queue import Empty
import time
import logging

from era_5g_interface.interface_helpers import LatencyMeasurements
from fcw.core.collision import *
from fcw.core.detection import *
from fcw.core.sort import Sort
from fcw.core.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class CollisionWorker(Thread):
    def __init__(
        self,
        image_queue: Queue,
        sio,
        config: dict,
        camera_config: dict,
        fps: float,
        **kw
    ):
        super().__init__(**kw)
        self.stop_event = Event()
        self.image_queue = image_queue
        self.sio = sio
        self.frame_id = 0
        self.latency_measurements = LatencyMeasurements()

        logger.info("Initializing object detector")
        self.detector = YOLODetector.from_dict(config.get("detector", {}))
        logger.info("Initializing image tracker")
        self.tracker = Sort.from_dict(config.get("tracker", {}))
        logger.info("Initializing forward collision guard")
        self.guard = ForwardCollisionGuard.from_dict(config.get("fcw", {}))
        self.guard.dt = 1 / fps
        logger.info("Initializing camera calibration")
        self.camera = Camera.from_dict(camera_config)

    def stop(self):
        self.stop_event.set()

    def __del__(self):
        logger.info("Delete object detector")
        del self.detector

    def run(self):
        """
        Periodically reads images from python internal queue process them.
        """

        logger.info(f"{self.name} thread is running.")

        while not self.stop_event.is_set():
            # Get image and metadata from input queue
            try:
                metadata, image = self.image_queue.get(block=True, timeout=1)
            except Empty:
                continue
            metadata["timestamp_before_process"] = time.perf_counter_ns()
            self.frame_id += 1
            # logger.info(f"Worker received frame id: {self.frame_id} {metadata['timestamp']}")
            try:
                detections = self.process_image(image)
                metadata["timestamp_after_process"] = time.perf_counter_ns()
                self.publish_results(detections, metadata)
            except Exception as e:
                logger.error(f"Exception with image processing: {repr(e)}")

        logger.info(f"{self.name} thread is stopping.")

    def process_image(self, image):
        # Detect object in image
        detections = self.detector.detect(image)
        # Get bounding boxes as numpy array
        detections = detections_to_numpy(detections)
        # Update state of image trackers
        self.tracker.update(detections)
        # Represent trackers as dict  tid -> KalmanBoxTracker
        tracked_objects = {
            t.id: t for t in self.tracker.trackers
            if t.hit_streak > self.tracker.min_hits and t.time_since_update < 1
        }
        # Get 3D locations of objects
        ref_points = get_reference_points(tracked_objects, self.camera, is_rectified=True)
        # Update state of objects in world
        self.guard.update(ref_points)

        return tracked_objects

    def publish_results(self, tracked_objects, metadata):
        """
        Publishes the results to the robot

        Args:
            tracked_objects (_type_): The results of the detection.
            metadata (_type_): NetApp-specific metadata related to processed image.
        """
        # Get list of current offenses
        dangerous_objects = self.guard.dangerous_objects()
        detections = dict()
        if tracked_objects is not None:
            for tid, t in tracked_objects.items():
                x1, y1, x2, y2 = t.get_state()[0]
                det = dict()
                det["bbox"] = [x1, y1, x2, y2]
                det["dangerous_distance"] = 0

                if tid in dangerous_objects.keys():
                    dist = Point(dangerous_objects[tid].location).distance(self.guard.vehicle_zone)
                    det["dangerous_distance"] = dist
                detections[tid] = det

                # det["class"] = result.label
                # det["class_name"] = self.detector.model.names[result.label]

            # TODO:check timestamp exists
            result = {"timestamp": metadata["timestamp"],
                      "recv_timestamp": metadata["recv_timestamp"],
                      "timestamp_before_process": metadata["timestamp_before_process"],
                      "timestamp_after_process": metadata["timestamp_after_process"],
                      "send_timestamp": time.perf_counter_ns(),
                      "detections": detections}
            self.sio.emit('message', result, namespace='/results', to=metadata["websocket_id"])
