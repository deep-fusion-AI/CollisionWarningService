from __future__ import annotations
import signal
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path
import cv2
import sys
import logging

from fcw.client_python.client_common import CollisionWarningClient
from fcw.client_python.client_common import StreamType
from era_5g_client.exceptions import FailedToConnect
from era_5g_interface.utils.rate_timer import RateTimer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("FCW client python")

stopped = False

# Video from source flag
FROM_SOURCE = True
# test video file
TEST_VIDEO_FILE = str("../../videos/video3.mp4")
# TEST_VIDEO_FILE = str("../../videos/bringauto_2023-03-20.mp4")

# Configuration of the algorithm
CONFIG_FILE = Path("../../config/config.yaml")
# Camera settings - specific for the particular input
CAMERA_CONFIG_FILE = Path("../../videos/video3.yaml")


# CAMERA_CONFIG_FILE = Path("../../videos/bringauto.yaml")


def main() -> None:
    """Creates the client class and starts the data transfer."""

    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--stream_type", type=int, help="StreamType: 1 = JPEG, 2 = H264",
        default=StreamType.H264
    )
    parser.add_argument("-c", "--config", type=Path, help="Collision warning config", default=CONFIG_FILE)
    parser.add_argument("--camera", type=Path, help="Camera settings", default=CAMERA_CONFIG_FILE)
    parser.add_argument("-o", "--out_csv_dir", type=str, help="Output CSV dir", default=".")
    parser.add_argument(
        "-p", "--out_prefix", type=str, help="Prefix of output csv file with measurements", default=None
    )
    parser.add_argument("-t", "--play_time", type=int, help="Video play time in seconds", default=5000)
    parser.add_argument("--fps", type=int, help="Video FPS", default=None)
    parser.add_argument("source_video", type=str, help="Video stream (file or url)", nargs='?', default=TEST_VIDEO_FILE)
    args = parser.parse_args()

    global stopped
    stopped = False
    collision_warning_client = None

    def signal_handler(sig: int, frame) -> None:
        logger.info(f"Terminating ({signal.Signals(sig).name})...")
        global stopped
        stopped = True
        #collision_warning_client.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # creates a video capture to pass images to the NetApp either from webcam ...
        logger.info(f"Opening video capture {args.source_video}")
        #cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/webcam.h264")
        cap = cv2.VideoCapture(args.source_video)
        if not cap.isOpened():
            raise Exception("Cannot open video capture")

        if not args.fps:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 60:
                fps = 30
        else:
            fps = args.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        logger.info(
            f"Video {width}x{height}, {fps} FPS, "
            f"{frames} frames, "
            f"{frames / fps} seconds"
        )

        logger.info(f"Stream type: {StreamType(args.stream_type)}")

        collision_warning_client = CollisionWarningClient(
            config=args.config, camera_config=args.camera, fps=fps, stream_type=StreamType(args.stream_type),
            out_csv_dir=args.out_csv_dir, out_prefix=args.out_prefix
        )

        rate_timer = RateTimer(rate=fps, time_function=time.perf_counter, iteration_miss_warning=True)
        start_time = time.perf_counter_ns()
        while time.perf_counter_ns() - start_time < args.play_time * 1.0e+9 and not stopped:
            ret, frame = cap.read()
            if not ret:
                break
            collision_warning_client.send_image(frame)
            rate_timer.sleep()  # sleep until next frame should be sent (with given fps)
        end_time = time.perf_counter_ns()
        logger.info(f"Total streaming time: {(end_time - start_time) * 1.0e-9:.3f}s")
        cap.release()

    except FailedToConnect as ex:
        logger.error(f"Failed to connect to server: {ex}")
    except KeyboardInterrupt:
        logger.info("Terminating...")
    except Exception as ex:
        traceback.print_exc()
        logger.error(f"Exception: {ex}")
    finally:
        if collision_warning_client is not None:
            collision_warning_client.stop()


if __name__ == "__main__":
    main()
