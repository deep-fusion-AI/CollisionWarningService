from __future__ import annotations

import logging
import os
import signal
import time
import traceback
from argparse import FileType, ArgumentParser
from pathlib import Path

import cv2

from fcw.client_python.client_common import CollisionWarningClient
from fcw.client_python.client_common import StreamType
from era_5g_client.exceptions import FailedToConnect

from fcw.core.rate_timer import RateTimer

stopped = False

# Video from source flag
FROM_SOURCE = False
# test video file
#TEST_VIDEO_FILE = str("../../videos/video3.mp4")
TEST_VIDEO_FILE = str("../../videos/bringauto_2023-03-20.mp4")

# Configuration of the algorithm
CONFIG_FILE = Path("../../config/config.yaml")
# Camera settings - specific for the particular input
#CAMERA_CONFIG_FILE = Path("../../videos/video3.yaml")
CAMERA_CONFIG_FILE = Path("../../videos/bringauto.yaml")


def main() -> None:
    """Creates the client class and starts the data transfer."""

    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--stream_type", type=int, help="StreamType: 1 = GSTREAMER, 2 = WEBSOCKETS, 3 = HTTP",
        default=StreamType.GSTREAMER
    )
    parser.add_argument("-c", "--config", type=Path, help="Collision warning config", default=CONFIG_FILE)
    parser.add_argument("--camera", type=Path, help="Camera settings", default=CAMERA_CONFIG_FILE)
    parser.add_argument("-o", "--out_csv_dir", type=str, help="Output CSV dir", default=".")
    parser.add_argument("-p", "--out_prefix", type=str, help="Prefix of output csv file with measurements", default=None)
    parser.add_argument("-t", "--play_time", type=int, help="Video play time", default=10)
    parser.add_argument("--fps", type=int, help="Video FPS", default=None)
    parser.add_argument("source_video", type=str, help="Video stream (file or url)", nargs='?', default=TEST_VIDEO_FILE)
    args = parser.parse_args()

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
            logging.info("Opening video file {file}".format(file=args.source_video))
            cap = cv2.VideoCapture(args.source_video)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
        if not args.fps:
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            fps = args.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("Video {W}x{H}, {fps} FPS".format(W=width, H=height, fps=fps))

        logging.info(f"Stream type: {StreamType(args.stream_type)}")

        # gstreamer True or False
        collision_warning_client = CollisionWarningClient(
            config=args.config, camera_config=args.camera, fps=fps, stream_type=StreamType(args.stream_type),
            out_csv_dir=args.out_csv_dir, out_prefix=args.out_prefix
        )

        logging.info(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        rate_timer = RateTimer(rate=fps, iteration_miss_warning=True)

        start_time = time.time_ns()
        while time.time_ns() - start_time < args.play_time * 1.0e+9 and not stopped:
            time0 = time.time_ns()
            ret, frame = cap.read()
            if not ret:
                break
            collision_warning_client.send_image(frame)
            time1 = time.time_ns()
            time_elapsed_s = (time1 - time0) * 1.0e-9
            logging.info(f"send_image time: {time_elapsed_s:.3f}")
            # if time_elapsed_s < (1/fps/2):
            #    print(f"time.sleep: {(1/fps)-time_elapsed_s:.3f}")
            #    time.sleep((1/fps)-time_elapsed_s)
            rate_timer.sleep()  # sleep until next frame should be sent (with given fps)
        end_time = time.time_ns()
        logging.info(f"Total streaming time: {(end_time - start_time) * 1.0e-9:.3f}s")
        cap.release()

    except FailedToConnect as ex:
        logging.error(f"Failed to connect to server ({ex})")
    except KeyboardInterrupt:
        logging.info("Terminating...")
    except Exception as ex:
        traceback.print_exc()
        logging.error(f"Failed to create client instance ({ex})")
    finally:
        if collision_warning_client is not None:
            collision_warning_client.stop()


if __name__ == "__main__":
    main()
