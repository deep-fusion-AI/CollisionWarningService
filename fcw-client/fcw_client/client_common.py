from __future__ import annotations

import csv
import logging
import os
import statistics
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Callable, Optional
import numpy as np
import yaml

from fcw_core_utils.geometry import Camera
from era_5g_client.client_base import NetAppClientBase

logger = logging.getLogger(__name__)

image_storage: Dict[int, np.ndarray] = dict()

DEBUG_PRINT_WARNING = True  # Prints score.
DEBUG_PRINT_DELAY = True  # Prints the delay between capturing image and receiving the results.

# URL of the FCW service
NETAPP_ADDRESS = str(os.getenv("NETAPP_ADDRESS", "http://localhost:5896"))


class ResultsReader:
    """Default class for processing FCW results."""

    def __init__(self, out_csv_dir: str = None, out_prefix: str = None) -> None:
        """Constructor.

        Args:
            out_csv_dir (str, optional): Dir for CSV timestamps stats.
            out_prefix (str, optional): Filename prefix for CSV timestamps stats.
        """

        self.delays = []
        self.delays_recv = []
        self.delays_send = []
        self.timestamps = [
            ["start_timestamp_ns",
             "recv_timestamp_ns",
             "send_timestamp_ns",
             "end_timestamp_ns"]
        ]
        self.out_csv_dir = out_csv_dir
        self.out_prefix = out_prefix

    def stats(self, send_frames_count: int) -> None:
        """Print timestamps stats and can write them to CSV file.

        Args:
            send_frames_count (int): Number of send frames.
        """

        logger.info(f"-----")
        if len(self.delays) < 1 or len(self.delays_recv) < 1 or len(self.delays_send) < 1:
            logger.warning(f"No results data received")
        else:
            logger.info(f"Send frames: {send_frames_count}, dropped frames: {send_frames_count - len(self.delays)}")
            logger.info(
                f"Delay median: {statistics.median(self.delays) * 1.0e-9:.3f}s "
                f"mean: {statistics.mean(self.delays) * 1.0e-9:.3f}s "
                f"min: {min(self.delays) * 1.0e-9:.3f}s "
                f"max: {max(self.delays) * 1.0e-9:.3f}s"
            )
            # logger.info(
            #     f"Delay service recv median: {statistics.median(self.delays_recv) * 1.0e-9:.3f}s "
            #     f"mean: {statistics.mean(self.delays_recv) * 1.0e-9:.3f}s "
            #     f"min: {min(self.delays_recv) * 1.0e-9:.3f}s "
            #     f"max: {max(self.delays_recv) * 1.0e-9:.3f}s"
            # )
            # logger.info(
            #     f"Delay service send median: {statistics.median(self.delays_send) * 1.0e-9:.3f}s "
            #     f"mean: {statistics.mean(self.delays_send) * 1.0e-9:.3f}s "
            #     f"min: {min(self.delays_send) * 1.0e-9:.3f}s "
            #     f"max: {max(self.delays_send) * 1.0e-9:.3f}s"
            # )
            if self.out_csv_dir is not None:
                out_csv_filename = f'{self.out_prefix}'
                out_csv_filepath = os.path.join(self.out_csv_dir, out_csv_filename + ".csv")
                with open(out_csv_filepath, "w", newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(self.timestamps)

    def get_results(self, results: Dict[str, Any]) -> None:
        """Callback which process the results from the FCW service.

        Args:
            results (Dict[str, Any]): The results in JSON format.
        """

        results_timestamp = time.perf_counter_ns()

        # Process dangerous detections
        if "dangerous_detections" in results:
            if DEBUG_PRINT_WARNING:
                for tracked_id, detection in results["dangerous_detections"].items():
                    score = float(detection["dangerous_distance"])
                    if score > 0:
                        logger.info(f"Dangerous distance {score:.2f}m to the object with id {tracked_id}")
        if "objects" in results:
            logger.info(f"objects {results['objects']}")

        # Process timestamps
        if "timestamp" in results:
            timestamp = results["timestamp"]
            recv_timestamp = results["recv_timestamp"]
            send_timestamp = results["send_timestamp"]

            if DEBUG_PRINT_DELAY:
                logger.info(
                    f"Result number {len(self.timestamps)}"
                    f", delay: {(results_timestamp - timestamp) * 1.0e-9:.3f}s"
                    # f", recv frame delay: {(recv_timestamp - timestamp) * 1.0e-9:.3f}s"
                )
                self.delays.append((results_timestamp - timestamp))
                self.delays_recv.append((recv_timestamp - timestamp))
                self.delays_send.append((send_timestamp - timestamp))

            self.timestamps.append(
                [
                    timestamp,
                    recv_timestamp,
                    send_timestamp,
                    results_timestamp
                ]
            )


class StreamType(Enum):
    JPEG = 1
    H264 = 2


class CollisionWarningClient:
    """Wrapper class for FCW client."""

    def __init__(
        self,
        config: Path,
        camera_config: Path,
        netapp_address: str = NETAPP_ADDRESS,
        fps: float = 30,
        viz: bool = True,
        viz_zmq_port: int = 5558,
        results_callback: Optional[Callable] = None,
        stream_type: Optional[StreamType] = StreamType.H264,
        out_csv_dir: Optional[str] = None,
        out_prefix: Optional[str] = "fcw_test_",
    ) -> None:
        """Constructor.

        Args:
            config (Path): Path to FCW configuration file.
            camera_config (Path): Path to camera configuration file.
            netapp_address (str, optional): The URI and port of the FCW service interface.
                Default taken from environment variables NETAPP_ADDRESS and NETAPP_PORT.
            fps (float, optional): Video FPS. Default to 30.
            results_callback (Callable, optional): Callback for receiving results.
                Default to ResultsReader.get_results
            stream_type (StreamType, optional): Stream type JPEG or H264. Default to H264.
            out_csv_dir (str, optional): Dir for CSV timestamps stats. Default to None.
            out_prefix (str, optional): Filename prefix for CSV timestamps stats. Default to "fcw_test_".
        """

        logger.info("Loading configuration file {cfg}".format(cfg=config))
        self.config_dict = yaml.safe_load(config.open())
        logger.info("Loading camera configuration {cfg}".format(cfg=camera_config))
        self.camera_config_dict = yaml.safe_load(camera_config.open())
        logger.info("Initializing camera calibration")
        self.camera = Camera.from_dict(self.camera_config_dict)

        width, height = self.camera.rectified_size
        self.fps = fps
        # Check bad loaded FPS
        if self.fps > 60:
            logger.warning(f"FPS {self.fps} is strangely high, newly set to 30")
            self.fps = 30
        self.results_callback = results_callback
        if self.results_callback is None:
            self.results_viewer = ResultsReader(
                out_csv_dir=out_csv_dir, out_prefix=out_prefix
            )
            self.results_callback = self.results_viewer.get_results
        self.stream_type = stream_type
        self.frame_id = 0

        # Create FCW client
        self.client = NetAppClientBase(self.results_callback)
        logger.info(f"Register with netapp_address: {netapp_address}")
        # Register client
        try:
            if self.stream_type is StreamType.H264:
                self.client.register(
                    netapp_address,
                    args={"h264": True, "config": self.config_dict, "camera_config": self.camera_config_dict,
                          "fps": self.fps, "viz": viz, "viz_zmq_port": viz_zmq_port,
                          "width": width, "height": height}
                )
            elif self.stream_type is StreamType.JPEG:
                self.client.register(
                    netapp_address,
                    args={"config": self.config_dict, "camera_config": self.camera_config_dict,
                          "fps": self.fps, "viz": viz, "viz_zmq_port": viz_zmq_port}
                )
            else:
                raise Exception("Unknown stream type")
        except Exception as e:
            self.client.disconnect()
            raise e
        logger.info(f"Client registered")

    def send_image(self, frame: np.ndarray, timestamp: Optional[int] = None) -> None:
        """Send image to FCW service including rectification.

        Args:
            frame (np.ndarray): Image in numpy array format ("bgr24").
            timestamp (int, optional): Timestamp for frame and results synchronization.
                Default to current time (time.perf_counter_ns())
        """

        if self.client is not None:
            self.frame_id += 1
            frame_undistorted = self.camera.rectify_image(frame)
            if not timestamp:
                timestamp = time.perf_counter_ns()
            self.client.send_image_ws(frame_undistorted, timestamp)

    def stop(self) -> None:
        """Print stats and disconnect from FCW service."""
        logger.info("Collision warning client stopping")

        if hasattr(self, "results_viewer") and self.results_viewer is not None:
            self.results_viewer.stats(self.frame_id)
        if self.client is not None:
            self.client.disconnect()
