from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, Callable, Optional

import cv2
import numpy as np
import yaml

from era_5g_client.client_base import NetAppClientBase
from era_5g_client.data_sender_gstreamer import DataSenderGStreamer
from era_5g_client.dataclasses import NetAppLocation
from fcw.core.geometry import Camera

image_storage: Dict[str, np.ndarray] = dict()
timestamps: Dict[str, int] = dict()
results_storage: Queue[Dict[str, Any]] = Queue()

DEBUG_PRINT_SCORE = True  # prints score
DEBUG_PRINT_DELAY = True  # prints the delay between capturing image and receiving the results

# ip address or hostname of the computer, where the netapp is deployed
NETAPP_ADDRESS = os.getenv("NETAPP_ADDRESS", "127.0.0.1")
# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)


class ResultsViewer(Thread):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.stop_event = Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        logging.info("Thread %s: starting", self.name)
        while not self.stop_event.is_set():
            if not results_storage.empty():
                results = results_storage.get(timeout=1)
                timestamp_str = results["timestamp"]
                timestamp = int(timestamp_str)
                if timestamp_str in timestamps:
                    timestamp = timestamps.pop(timestamp_str)
                if DEBUG_PRINT_DELAY:
                    time_now = results["results_timestamp"]
                    print(f"Delay: {(time_now - timestamp) * 1.0e-9:.3f}s ")
                try:
                    frame = image_storage.pop(timestamp_str)
                    detections = results["detections"]
                    for d in detections:
                        score = float(d["score"])
                        if DEBUG_PRINT_SCORE and score > 0:
                            print(f"Score: {score}")
                        # cls_name = d["class_name"]
                        # Draw detection into frame.
                        x1, y1, x2, y2 = [int(coord) for coord in d["bbox"]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if score > 0:
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), cv2.FILLED)
                            frame = cv2.addWeighted(overlay, 0.3, frame, 1 - 0.3, 0)
                            cv2.putText(
                                frame,
                                # f"{cls_name} ({score * 100:.0f})%",
                                f"{score:.1f} m",
                                (x1, y1 - 5),
                                font,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    try:
                        cv2.imshow("Results", frame)
                        cv2.waitKey(1)
                    except Exception as ex:
                        print(ex)
                    results_storage.task_done()
                except KeyError as ex:
                    print(f"image_storage KeyError {ex}")


def get_results(results: Dict[str, Any]) -> None:
    """Callback which process the results from the NetApp.

    Args:
        results (str): The results in json format
    """
    results_timestamp = time.time_ns()
    print(results)
    if "timestamp" in results:
        results_storage.put(dict(results, **{"results_timestamp": results_timestamp}), block=False)


class CollisionWarningClient:

    def __init__(
        self,
        config: Path = None,
        camera_config: Path = None,
        fps: float = 30,
        results_callback: Callable = None,
        gstreamer: bool = False
    ):
        logging.info("Loading configuration file {cfg}".format(cfg=config))
        self.config_dict = yaml.safe_load(config.open())
        logging.info("Loading camera configuration {cfg}".format(cfg=camera_config))
        self.camera_config_dict = yaml.safe_load(camera_config.open())
        logging.info("Initializing camera calibration")
        self.camera = Camera.from_dict(self.camera_config_dict)
        width, height = self.camera.rectified_size
        self.fps = fps
        self.results_callback = results_callback
        if self.results_callback is None:
            self.results_callback = get_results
            self.results_viewer = ResultsViewer(name="results_viewer", daemon=True)
            self.results_viewer.start()
        self.gstreamer = gstreamer
        self.frame_id = 0

        self.client = NetAppClientBase(self.results_callback)

        if self.gstreamer:
            # register the client with the NetApp with gstreamer extension
            self.client.register(
                NetAppLocation(NETAPP_ADDRESS, NETAPP_PORT),
                gstreamer=True,
                args={"config": self.config_dict, "camera_config": self.camera_config_dict, "fps": self.fps}
            )
            if not self.client.gstreamer_port:
                logging.error("Missing port for GStreamer")
                self.client.disconnect()
                return
            # TODO: check fps
            self.data_sender_gstreamer = DataSenderGStreamer(
                self.client.netapp_location.address, self.client.gstreamer_port, self.fps, width, height
            )
        else:
            # register the client with the NetApp  without gstreamer extension
            self.client.register(
                NetAppLocation(NETAPP_ADDRESS, NETAPP_PORT),
                args={"config": self.config_dict, "camera_config": self.camera_config_dict, "fps": self.fps}
            )

    def send_image(self, frame: np.ndarray, timestamp: Optional[str] = None):
        if not timestamp:
            timestamp = time.time_ns()
        frame_undistorted = self.camera.rectify_image(frame)
        timestamp_str = str(timestamp)
        # TODO: can overflow?
        self.frame_id += 1
        # TODO: timestamp with gstreamer
        if self.gstreamer:
            self.data_sender_gstreamer.send_image(frame_undistorted)
            if self.results_callback is get_results:
                image_storage[str(self.frame_id)] = frame_undistorted
                timestamps[str(self.frame_id)] = timestamp
        else:
            self.client.send_image_http(frame_undistorted, timestamp_str, 5)
            if self.results_callback is get_results:
                image_storage[timestamp_str] = frame_undistorted

    def stop(self):
        if self.results_viewer is not None:
            self.results_viewer.stop()
        if self.client is not None:
            self.client.disconnect()
        if self.gstreamer:
            self.data_sender_gstreamer.out.release()
