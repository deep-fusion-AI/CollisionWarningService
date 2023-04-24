from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, Callable, Optional
from enum import Enum
import statistics
from datetime import datetime
import csv

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

start_timestamp = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")


class ResultsViewer(Thread):
    def __init__(self, out_csv_dir: str = None, out_prefix: str = None, **kw) -> None:
        super().__init__(**kw)
        self.stop_event = Event()
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

    def stop(self) -> None:
        self.stop_event.set()
        logging.info(f"-----")
        if len(self.delays) < 1 or len(self.delays_recv) < 1 or len(self.delays_send) < 1:
            logging.warning(f"No results data received")
        else:
            logging.info(f"Delay median: {statistics.median(self.delays) * 1.0e-9:.3f}s")
            logging.info(f"Delay service recv median: {statistics.median(self.delays_recv) * 1.0e-9:.3f}s")
            logging.info(f"Delay service send median: {statistics.median(self.delays_send) * 1.0e-9:.3f}s")
            if self.out_csv_dir is not None:
                out_csv_filename = f'{start_timestamp}_{self.out_prefix}'
                out_csv_filepath = os.path.join(self.out_csv_dir, out_csv_filename + ".csv")
                with open(out_csv_filepath, "w", newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(self.timestamps)

    def run(self) -> None:
        logging.info("Thread %s: starting", self.name)
        while not self.stop_event.is_set():
            if not results_storage.empty():
                results = results_storage.get(timeout=1)
                timestamp_str = results["timestamp"]
                recv_timestamp_str = results["recv_timestamp"]
                send_timestamp_str = results["send_timestamp"]
                timestamp = int(timestamp_str)
                recv_timestamp = int(recv_timestamp_str)
                send_timestamp = int(send_timestamp_str)
                if timestamp_str in timestamps:
                    timestamp = timestamps.pop(timestamp_str)
                    logging.info(f"Recv frame id: {timestamp_str}")
                if DEBUG_PRINT_DELAY:
                    time_now = results["results_timestamp"]
                    logging.info(f"Delay: {(time_now - timestamp) * 1.0e-9:.3f}s ")
                    logging.info(f"Delay service recv: {(recv_timestamp - timestamp) * 1.0e-9:.3f}s ")
                    logging.info(f"Delay service send: {(send_timestamp - timestamp) * 1.0e-9:.3f}s ")
                    self.delays.append((time_now - timestamp))
                    self.delays_recv.append((recv_timestamp - timestamp))
                    self.delays_send.append((send_timestamp - timestamp))

                self.timestamps.append(
                    [
                        timestamp,
                        recv_timestamp,
                        send_timestamp,
                        results["results_timestamp"]
                    ]
                )

                try:
                    frame = image_storage.pop(timestamp_str)
                    detections = results["detections"]
                    for d in detections:
                        score = float(d["score"])
                        if DEBUG_PRINT_SCORE and score > 0:
                            logging.info(f"Score: {score}")
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
                        logging.debug(ex)
                    results_storage.task_done()
                except KeyError as ex:
                    logging.error(f"image_storage KeyError {ex}")


def get_results(results: Dict[str, Any]) -> None:
    """Callback which process the results from the NetApp.

    Args:
        results (str): The results in json format
    """
    results_timestamp = time.time_ns()
    # print(results)
    if "timestamp" in results:
        results_storage.put(dict(results, **{"results_timestamp": results_timestamp}), block=False)


class StreamType(Enum):
    GSTREAMER = 1
    WEBSOCKETS = 2
    HTTP = 3


class CollisionWarningClient:

    def __init__(
        self,
        config: Path = None,
        camera_config: Path = None,
        fps: float = 30,
        results_callback: Optional[Callable] = None,
        stream_type: Optional[StreamType] = StreamType.HTTP,
        out_csv_dir: Optional[str] = None,
        out_prefix: Optional[str] = "fcw_test_",
        netapp_address: Optional[str] = NETAPP_ADDRESS,
        netapp_port: Optional[int] = NETAPP_PORT
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
            self.results_viewer = ResultsViewer(
                name="results_viewer", out_csv_dir=out_csv_dir, out_prefix=out_prefix, daemon=True
            )
            self.results_viewer.start()
        self.stream_type = stream_type
        self.frame_id = 0

        self.client = NetAppClientBase(self.results_callback)
        logging.info(f"Register with netapp_address: {netapp_address}, netapp_port: {netapp_port}")

        if self.stream_type is StreamType.GSTREAMER:
            # register the client with the NetApp with gstreamer extension
            self.client.register(
                NetAppLocation(netapp_address, netapp_port),
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
        elif self.stream_type is StreamType.WEBSOCKETS:
            # register the client with the NetApp  without gstreamer extension
            self.client.register(
                NetAppLocation(netapp_address, netapp_port),
                ws_data=True,
                args={"config": self.config_dict, "camera_config": self.camera_config_dict, "fps": self.fps}
            )
        elif self.stream_type is StreamType.HTTP:
            # register the client with the NetApp  without gstreamer extension
            self.client.register(
                NetAppLocation(netapp_address, netapp_port),
                args={"config": self.config_dict, "camera_config": self.camera_config_dict, "fps": self.fps}
            )
        else:
            raise Exception("Unknown stream type")

    def send_image(self, frame: np.ndarray, timestamp: Optional[str] = None):
        self.frame_id += 1
        time0 = time.time_ns()
        frame_undistorted = self.camera.rectify_image(frame)
        time1 = time.time_ns()
        time_elapsed_s = (time1 - time0) * 1.0e-9
        logging.info(f"rectify_image time: {time_elapsed_s:.3f}")
        if not timestamp:
            timestamp = time.time_ns()
        timestamp_str = str(timestamp)
        if self.stream_type is StreamType.GSTREAMER:
            self.data_sender_gstreamer.send_image(frame_undistorted)
            if self.results_callback is get_results:
                image_storage[str(self.frame_id)] = frame_undistorted
                timestamps[str(self.frame_id)] = timestamp
                logging.info(f"Sent frame id: {self.frame_id}")
        elif self.stream_type is StreamType.WEBSOCKETS:
            self.client.send_image_ws(frame_undistorted, timestamp_str)
            if self.results_callback is get_results:
                image_storage[timestamp_str] = frame_undistorted
        elif self.stream_type is StreamType.HTTP:
            self.client.send_image_http(frame_undistorted, timestamp_str, 5)
            if self.results_callback is get_results:
                image_storage[timestamp_str] = frame_undistorted
        else:
            raise Exception("Unknown stream type")

    def frame_id(self):
        return self.frame_id

    def stop(self):
        if self.results_viewer is not None:
            self.results_viewer.stop()
        if self.client is not None:
            self.client.disconnect()
        if self.stream_type is StreamType.GSTREAMER:
            self.data_sender_gstreamer.out.release()
