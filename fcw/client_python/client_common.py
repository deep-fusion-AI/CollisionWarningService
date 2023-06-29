from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Callable, Optional
from enum import Enum
import statistics
from datetime import datetime
import csv
import numpy as np
import yaml

from era_5g_client.client_base import NetAppClientBase
from era_5g_client.dataclasses import NetAppLocation

from fcw.core.geometry import Camera

logger = logging.getLogger(__name__)

image_storage: Dict[int, np.ndarray] = dict()

DEBUG_PRINT_SCORE = True  # prints score
DEBUG_PRINT_DELAY = True  # prints the delay between capturing image and receiving the results

# ip address or hostname of the computer, where the netapp is deployed
NETAPP_ADDRESS = os.getenv("NETAPP_ADDRESS", "127.0.0.1")

# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)

start_timestamp = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")


class ResultsViewer:
    def __init__(self, out_csv_dir: str = None, out_prefix: str = None) -> None:
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

    def stats(self, send_frames_count) -> None:
        logger.info(f"-----")
        if len(self.delays) < 1 or len(self.delays_recv) < 1 or len(self.delays_send) < 1:
            logger.warning(f"No results data received")
        else:
            logger.info(f"Dropped frames: {send_frames_count - len(self.delays)}, (send frames: {send_frames_count})")
            logger.info(
                f"Delay              median: {statistics.median(self.delays) * 1.0e-9:.3f}s "
                f"mean: {statistics.mean(self.delays) * 1.0e-9:.3f}s "
                f"min: {min(self.delays) * 1.0e-9:.3f}s "
                f"max: {max(self.delays) * 1.0e-9:.3f}s"
            )
            logger.info(
                f"Delay service recv median: {statistics.median(self.delays_recv) * 1.0e-9:.3f}s "
                f"mean: {statistics.mean(self.delays_recv) * 1.0e-9:.3f}s "
                f"min: {min(self.delays_recv) * 1.0e-9:.3f}s "
                f"max: {max(self.delays_recv) * 1.0e-9:.3f}s"
            )
            logger.info(
                f"Delay service send median: {statistics.median(self.delays_send) * 1.0e-9:.3f}s "
                f"mean: {statistics.mean(self.delays_send) * 1.0e-9:.3f}s "
                f"min: {min(self.delays_send) * 1.0e-9:.3f}s "
                f"max: {max(self.delays_send) * 1.0e-9:.3f}s"
            )
            if self.out_csv_dir is not None:
                out_csv_filename = f'{self.out_prefix}'
                out_csv_filepath = os.path.join(self.out_csv_dir, out_csv_filename + ".csv")
                with open(out_csv_filepath, "w", newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(self.timestamps)

    def get_results(self, results: Dict[str, Any]) -> None:
        """Callback which process the results from the NetApp.

        Args:
            results (str): The results in json format
        """
        results_timestamp = time.perf_counter_ns()
        if "timestamp" in results:
            timestamp = results["timestamp"]
            recv_timestamp = results["recv_timestamp"]
            send_timestamp = results["send_timestamp"]
            if DEBUG_PRINT_DELAY:
                logger.info(f" {len(self.timestamps)} Delay: {(results_timestamp - timestamp) * 1.0e-9:.3f}s ")
                # logger.info(f"Delay service recv: {(recv_timestamp - timestamp) * 1.0e-9:.3f}s ")
                # logger.info(f"Delay service send: {(send_timestamp - timestamp) * 1.0e-9:.3f}s ")
                self.delays.append((results_timestamp - timestamp))
                self.delays_recv.append((recv_timestamp - timestamp))
                self.delays_send.append((send_timestamp - timestamp))

            if DEBUG_PRINT_SCORE:
                detections = results["detections"]
                for d in detections:
                    score = float(d["score"])
                    if score > 0:
                        logger.info(f"Score: {score}")

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

    def __init__(
        self,
        config: Path = None,
        camera_config: Path = None,
        fps: float = 30,
        results_callback: Optional[Callable] = None,
        stream_type: Optional[StreamType] = StreamType.H264,
        out_csv_dir: Optional[str] = None,
        out_prefix: Optional[str] = "fcw_test_",
        netapp_address: Optional[str] = NETAPP_ADDRESS,
        netapp_port: Optional[int] = NETAPP_PORT
    ):
        logger.info("Loading configuration file {cfg}".format(cfg=config))
        self.config_dict = yaml.safe_load(config.open())
        logger.info("Loading camera configuration {cfg}".format(cfg=camera_config))
        self.camera_config_dict = yaml.safe_load(camera_config.open())
        logger.info("Initializing camera calibration")
        self.camera = Camera.from_dict(self.camera_config_dict)
        width, height = self.camera.rectified_size
        self.fps = fps
        self.results_callback = results_callback
        if self.results_callback is None:
            self.results_viewer = ResultsViewer(
                out_csv_dir=out_csv_dir, out_prefix=out_prefix
            )
            self.results_callback = self.results_viewer.get_results
        self.stream_type = stream_type
        self.frame_id = 0

        self.client = NetAppClientBase(self.results_callback)
        logger.info(f"Register with netapp_address: {netapp_address}, netapp_port: {netapp_port}")

        if self.stream_type is StreamType.H264:
            self.client.register(
                NetAppLocation(netapp_address, netapp_port),
                args={"h264": True, "config": self.config_dict, "camera_config": self.camera_config_dict,
                      "fps": self.fps,
                      "width": width, "height": height}
            )
        elif self.stream_type is StreamType.JPEG:
            self.client.register(
                NetAppLocation(netapp_address, netapp_port),
                args={"config": self.config_dict, "camera_config": self.camera_config_dict, "fps": self.fps}
            )
        else:
            raise Exception("Unknown stream type")

    def send_image(self, frame: np.ndarray, timestamp: Optional[int] = None):
        if self.client is not None:
            self.frame_id += 1
            frame_undistorted = self.camera.rectify_image(frame)
            if not timestamp:
                timestamp = time.perf_counter_ns()
            self.client.send_image_ws(frame_undistorted, timestamp)

    def stop(self):
        if hasattr(self, "results_viewer") and self.results_viewer is not None:
            self.results_viewer.stats(self.frame_id)
        if self.client is not None:
            self.client.disconnect()
