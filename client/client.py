from __future__ import annotations

import logging
import math
import os
import signal
import time
import traceback
from queue import Queue
from threading import Event, Thread
from types import FrameType
from typing import Any, Dict, Optional
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.geometry import Camera

from era_5g_client.client import NetAppClient
from era_5g_client.exceptions import FailedToConnect

image_storage: Dict[str, np.ndarray] = dict()
results_storage: Queue[Dict[str, Any]] = Queue()
stopped = False

DEBUG_PRINT_SCORE = False  # useful for FPS detector
DEBUG_PRINT_DELAY = False  # prints the delay between capturing image and receiving the results

# Video from source flag
FROM_SOURCE = False
# ip address or hostname of the computer, where the netapp is deployed
NETAPP_ADDRESS = os.getenv("NETAPP_ADDRESS", "127.0.0.1")
# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)
# test video file
TEST_VIDEO_FILE = os.getenv("TEST_VIDEO_FILE", "../videos/video3.mp4")

# Configuration of the algorithm
config = Path("../config/config.yaml")
# Camera settings - specific for the particular input
camera_config = Path("../videos/video3.yaml")

camera = None


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
                if DEBUG_PRINT_DELAY:
                    time_now = math.floor(time.time() * 100)
                    print(f"{(time_now - timestamp) / 100.}s delay")
                try:
                    frame = image_storage.pop(timestamp_str)
                    detections = results["detections"]
                    for d in detections:
                        score = float(d["score"])
                        if DEBUG_PRINT_SCORE and score > 0:
                            print(score)
                        #cls_name = d["class_name"]
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
                                #f"{cls_name} ({score * 100:.0f})%",
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
                    print(ex)


def get_results(results: Dict[str, Any]) -> None:
    """Callback which process the results from the NetApp

    Args:
        results (str): The results in json format
    """

    print(results)
    if "timestamp" in results:
        results_storage.put(results, block=False)
    pass


def main() -> None:
    """Creates the client class and starts the data transfer."""

    results_viewer = ResultsViewer(name="test_client_http_viewer", daemon=True)
    results_viewer.start()

    logging.getLogger().setLevel(logging.INFO)

    client = None
    global stopped
    stopped = False

    def signal_handler(sig: int, frame: Optional[FrameType]) -> None:
        global stopped
        stopped = True
        results_viewer.stop()
        print(f"Terminating ({signal.Signals(sig).name})...")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    global camera

    try:
        logging.info("Loading configuration file {cfg}".format(cfg=config))
        config_dict = yaml.safe_load(config.open())
        logging.info("Loading camera configuration {cfg}".format(cfg=camera_config))
        camera_config_dict = yaml.safe_load(camera_config.open())
        logging.info("Initializing camera calibration")
        camera = Camera.from_dict(camera_config_dict)

        if FROM_SOURCE:
            # creates a video capture to pass images to the NetApp either from webcam ...
            cap = cv2.VideoCapture(0)
            logging.info("Opening webcam")
            if not cap.isOpened():
                raise Exception("Cannot open camera")
        else:
            # or from video file
            logging.info("Opening video file {file}".format(file=TEST_VIDEO_FILE))
            cap = cv2.VideoCapture(TEST_VIDEO_FILE)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
        fps = cap.get(cv2.CAP_PROP_FPS)

        # creates the NetApp client with gstreamer extension
        client = NetAppClient("", "", "", "", True, get_results, False, False, NETAPP_ADDRESS, NETAPP_PORT)
        # register the client with the NetApp
        client.register({"config": config_dict, "camera_config": camera_config_dict, "fps": fps})

        while not stopped:
            ret, frame = cap.read()
            timestamp = math.floor(time.time() * 100)
            if not ret:
                break
            frame_undistorted = camera.rectify_image(frame)
            timestamp_str = str(timestamp)
            image_storage[timestamp_str] = frame_undistorted
            client.send_image(frame_undistorted, timestamp_str, 1)

    except FailedToConnect as ex:
        print(f"Failed to connect to server ({ex})")
    except KeyboardInterrupt:
        print("Terminating...")
    except Exception as ex:
        traceback.print_exc()
        print(f"Failed to create client instance ({ex})")
    finally:
        results_viewer.stop()
        if client is not None:
            client.disconnect()


if __name__ == "__main__":
    main()
