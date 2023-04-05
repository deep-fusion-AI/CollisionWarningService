from __future__ import annotations

import logging
import os
import signal
import time
import traceback
from pathlib import Path

import cv2

from client_common import CollisionWarningClient
from client_common import StreamType
from era_5g_client.exceptions import FailedToConnect

stopped = False

# Video from source flag
FROM_SOURCE = False
# test video file
TEST_VIDEO_FILE = os.getenv("TEST_VIDEO_FILE", "../../videos/video3.mp4")

# Configuration of the algorithm
config = Path("../../config/config.yaml")
# Camera settings - specific for the particular input
camera_config = Path("../../videos/video3.yaml")


def main() -> None:
    """Creates the client class and starts the data transfer."""

    logging.getLogger().setLevel(logging.INFO)

    collision_warning_client = None
    global stopped
    stopped = False

    def signal_handler(sig: int) -> None:
        global stopped
        stopped = True
        collision_warning_client.stop()
        print(f"Terminating ({signal.Signals(sig).name})...")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
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

        # gstreamer True or False
        collision_warning_client = CollisionWarningClient(
            config=config, camera_config=camera_config, fps=fps, stream_type=StreamType.WEBSOCKETS
        )

        print(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        start_time = time.time_ns()
        while not stopped:
            time0 = time.time_ns()
            ret, frame = cap.read()
            if not ret:
                break
            collision_warning_client.send_image(frame)
            time1 = time.time_ns()
            time_elapsed_s = (time1 - time0) * 1.0e-9
            print(f"send_image time: {time_elapsed_s:.3f}")
            #if time_elapsed_s < (1/fps/2):
            #    print(f"time.sleep: {(1/fps)-time_elapsed_s:.3f}")
            #    time.sleep((1/fps)-time_elapsed_s)
        end_time = time.time_ns()
        print(f"Total streaming time: {(end_time - start_time) * 1.0e-9:.3f}s")
        cap.release()

    except FailedToConnect as ex:
        print(f"Failed to connect to server ({ex})")
    except KeyboardInterrupt:
        print("Terminating...")
    except Exception as ex:
        traceback.print_exc()
        print(f"Failed to create client instance ({ex})")
    finally:
        collision_warning_client.stop()


if __name__ == "__main__":
    main()
