import time
from dataclasses import asdict
from multiprocessing import Queue
from queue import Empty
from threading import Thread, Event
from typing import Callable, Any

import zmq
from zmq import ZMQError

from era_5g_interface.interface_helpers import LatencyMeasurements
from fcw_core.detection import *
from fcw_core.sort import Sort, KalmanBoxTracker
from fcw_core.yolo_detector import YOLODetector
from fcw_core_utils.collision import *

logger = logging.getLogger(__name__)


class CollisionWorker(Thread):
    """FCW worker. Reads data from passed queue, performs FCW processing and returns results using callback."""

    def __init__(
        self,
        image_queue: Queue,
        send_function: Callable[[Dict[str, Any]], None],
        config: Dict,
        camera_config: Dict,
        fps: float,
        viz: bool = False,
        viz_zmq_port: int = 5558,
        **kw,
    ) -> None:
        """Constructor.

        Args:
            image_queue (Queue): The queue with all to-be-processed images.
            send_function (Callable[[Dict], None]): Callback used to send results.
            config (Dict): FCW config.
            camera_config (Dict): Camera config.
            fps (float): Framerate.
            viz (bool): Enable visualization?
            viz_zmq_port (int): Visualization ZeroMQ port.
            **kw: Thread arguments.
        """

        super().__init__(**kw)

        self._stop_event = Event()
        self.image_queue = image_queue
        self._send_function = send_function
        self._frame_id = 0
        self.latency_measurements: LatencyMeasurements = LatencyMeasurements()
        self._viz = viz

        logger.info("Initializing object detector")
        self._detector = YOLODetector.from_dict(config.get("detector", {}))
        logger.info("Initializing image tracker")
        self._tracker = Sort.from_dict(config.get("tracker", {}))
        logger.info("Initializing forward collision guard")
        self._guard = ForwardCollisionGuard.from_dict(config.get("fcw", {}))
        self._guard.dt = 1 / fps
        logger.info("Initializing camera calibration")
        self._camera = Camera.from_dict(camera_config)
        self._config = dict(config=config, camera_config=camera_config)

        # Visualization stuff.
        if self._viz:
            try:
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.PUB)
                logger.info(f"Publishing visualization on ZeroMQ tcp://*:{viz_zmq_port}")
                self._socket.bind("tcp://*:%s" % viz_zmq_port)
            except ZMQError as e:
                logger.error(f"Visualization was disabled on this worker: {e}")
                self._viz = False

    def stop(self) -> None:
        """Set stop event to stop FCW worker."""

        self._stop_event.set()

    def __del__(self) -> None:
        logger.info("Delete object detector")
        del self._detector

    def run(self) -> None:
        """FWC worker loop. Periodically reads images from python internal queue process them."""

        logger.info(f"{self.name} thread is running.")

        while not self._stop_event.is_set():
            # Get image and metadata from input queue.
            metadata: Dict[str, Any]
            image: np.ndarray
            try:
                metadata, image = self.image_queue.get(block=True, timeout=1)
            except Empty:
                continue
            # Store timestamp before processing.
            metadata["timestamp_before_process"] = time.perf_counter_ns()
            self._frame_id += 1
            # logger.info(f"Worker received frame id: {self.frame_id} {metadata['timestamp']}")
            try:
                detections = self._process_image(image)
                # Store timestamp after processing.
                metadata["timestamp_after_process"] = time.perf_counter_ns()
                # Generate results.
                results = self._generate_results(detections, metadata)
                # Send results via the provided callback.
                self._send_function(results)

                if self._viz:
                    # If visualisation is enabled, send image with results over ZeroMQ.
                    self._send_image_with_results(image, results)

            except Exception as ex:
                logger.error(f"Exception with image processing ({type(ex)}): {repr(ex)}")

        logger.info(f"{self.name} thread is stopping.")

    def _process_image(self, image: np.ndarray) -> Dict[int, KalmanBoxTracker]:
        """Process image by FCW.

        Args:
            image (np.ndarray): Image to be processed.

        Returns:
            Dictionary of KalmanBoxTrackers.
        """

        # Detect object in image.
        detections = self._detector.detect(image)
        # Get bounding boxes as numpy array.
        detections = detections_to_numpy(detections)
        # Update state of image trackers.
        self._tracker.update(detections)
        # Represent trackers as dict, {tid: KalmanBoxTracker, ...}.
        tracked_objects: Dict[int, KalmanBoxTracker] = {
            t.id: t for t in self._tracker.trackers if t.hit_streak > self._tracker.min_hits and t.time_since_update < 1
        }
        # Get 3D locations of objects.
        ref_points = get_reference_points(tracked_objects, self._camera, is_rectified=True)
        # Update state of objects in world.
        self._guard.update(ref_points)

        return tracked_objects

    def _send_image_with_results(self, image: np.ndarray, results: Dict[str, Any]) -> None:
        """Send a numpy array with metadata.

        Args:
            image (np.ndarray): Image.
            results (Dict[str, Any]): FCW results.
        """

        md = dict(dtype=str(image.dtype), shape=image.shape, results=dict(results, config=self._config))
        self._socket.send_json(md, 0 | zmq.SNDMORE)
        self._socket.send(image, 0, copy=True, track=False)

    def _generate_results(
        self, tracked_objects: Dict[int, KalmanBoxTracker], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates the results.

        Args:
            tracked_objects (Dict[int, KalmanBoxTracker]): The results of the detection.
            metadata (Dict[str, Any]): 5G-ERA Network Application specific metadata related to processed image.

        Returns:
            Dictionary of results.
        """

        # Get list of current offenses.
        dangerous_objects = self._guard.dangerous_objects()
        dangerous_detections = dict()

        # Get object statuses.
        object_statuses = list(self._guard.label_objects(include_distant=False))

        if tracked_objects is not None:
            for tid, t in tracked_objects.items():
                x1, y1, x2, y2 = t.get_state()[0]
                det = dict()
                det["bbox"] = [x1, y1, x2, y2]
                det["dangerous_distance"] = 0
                det["age"] = t.age
                det["hit_streak"] = t.hit_streak
                det["class"] = t.label
                det["class_name"] = self._detector.model.names[t.label]

                if tid in dangerous_objects.keys():
                    dist = Point(dangerous_objects[tid].location).distance(self._guard.vehicle_zone)
                    det["dangerous_distance"] = dist
                dangerous_detections[tid] = det

            # Make object statuses serializable - convert from shapely types.
            for i in range(len(object_statuses)):
                object_statuses[i].location = object_statuses[i].location.coords[0]
                object_statuses[i].path = [pts for pts in object_statuses[i].path.coords]
            object_statuses = [asdict(object_status) for object_status in object_statuses]

            send_timestamp = time.perf_counter_ns()

            self.latency_measurements.store_latency(send_timestamp - metadata["recv_timestamp"])

            return {
                "timestamp": metadata.get("timestamp", 0),
                "recv_timestamp": metadata.get("recv_timestamp", 0),
                "timestamp_before_process": metadata["timestamp_before_process"],
                "timestamp_after_process": metadata["timestamp_after_process"],
                "send_timestamp": send_timestamp,
                "dangerous_detections": dangerous_detections,
                "objects": object_statuses,
            }
