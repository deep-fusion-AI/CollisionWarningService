from __future__ import annotations
import signal
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path
import cv2
import sys
import logging

# FCW and 5G-ERA stuff.
from fcw_client.client_common import CollisionWarningClient
from fcw_client.client_common import StreamType
from era_5g_client.exceptions import FailedToConnect
from era_5g_interface.utils.rate_timer import RateTimer

# Set logging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("FCW client python")

# Testing video file.
TEST_VIDEO_FILE = str("../../../videos/video3.mp4")
# TEST_VIDEO_FILE = str("rtsp://127.0.0.1:8554/webcam.h264")
# TEST_VIDEO_FILE = str("../../../videos/bringauto_2023-03-20.mp4")

# Testing configuration of the algorithm.
CONFIG_FILE = Path("../../../config/config.yaml")

# Testing camera settings - specific for the particular input
CAMERA_CONFIG_FILE = Path("../../../videos/video3.yaml")
# CAMERA_CONFIG_FILE = Path("../../../videos/bringauto.yaml")

# stopped flag for SIGTERM handler (stopping video frames sending).
stopped = False
collision_warning_client = None


def signal_handler(sig: int, *_) -> None:
    """Signal handler for SIGTERM and SIGINT."""

    logger.info(f"Terminating ({signal.Signals(sig).name}) ...")
    global stopped
    stopped = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main() -> None:
    """Example of FCW client."""

    # Parse arguments.
    parser = ArgumentParser()
    parser.add_argument("-s", "--stream_type", type=int, help="StreamType: 1 = JPEG, 2 = H264", default=StreamType.H264)
    parser.add_argument("-c", "--config", type=Path, help="Collision warning config", default=CONFIG_FILE)
    parser.add_argument("--camera", type=Path, help="Camera settings", default=CAMERA_CONFIG_FILE)
    parser.add_argument("-o", "--out_csv_dir", type=str, help="Output CSV dir", default=None)
    parser.add_argument(
        "-p", "--out_prefix", type=str, help="Prefix of output csv file with measurements", default=None
    )
    parser.add_argument("-t", "--play_time", type=int, help="Video play time in seconds", default=5000)
    parser.add_argument("--fps", type=int, help="Video FPS", default=None)
    parser.add_argument("source_video", type=str, help="Video stream (file or url)", nargs="?", default=TEST_VIDEO_FILE)
    args = parser.parse_args()

    global collision_warning_client

    try:
        # Create a video capture to pass images to the 5G-ERA Network Application.
        logger.info(f"Opening video capture {args.source_video}")
        cap = cv2.VideoCapture(args.source_video)
        if not cap.isOpened():
            raise Exception("Cannot open video capture")

        # Check bad loaded FPS
        if not args.fps:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 60:
                logger.warning(f"FPS {fps} is strangely high, newly set to 30")
                fps = 30
        else:
            fps = args.fps

        # Get width, height and number of frames.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Number of frames can be wrong.
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        logger.info(f"Video {width}x{height}, {fps} FPS, " f"{frames} frames, " f"{frames / fps} seconds")
        logger.info(f"Stream type: {StreamType(args.stream_type)}")

        # Create collision warning client with given parameters,
        # width and height for client registration is loaded from camera config.
        collision_warning_client = CollisionWarningClient(
            config=args.config,
            camera_config=args.camera,
            fps=fps,
            stream_type=StreamType(args.stream_type),
            out_csv_dir=args.out_csv_dir,
            out_prefix=args.out_prefix,
        )

        # Rate timer for control the speed of a loop (fps).
        rate_timer = RateTimer(rate=fps, time_function=time.perf_counter, iteration_miss_warning=True)
        # Start time
        start_time = time.perf_counter_ns()
        # Check elapsed time or stopped flag.
        # Play time in ns
        play_time_ns = args.play_time * 1.0e9
        # Main processing loop
        logger.info("Start sending images ...")
        while time.perf_counter_ns() - start_time < play_time_ns and not stopped:
            # Read single frame from a stream.
            ret, frame = cap.read()
            if not ret:
                break
            # Send to FCW service for detection/processing.
            collision_warning_client.send_image(frame)
            # Sleep until next frame should be sent (with given fps).
            rate_timer.sleep()
        end_time = time.perf_counter_ns()
        logger.info(f"Total streaming time: {(end_time - start_time) * 1.0e-9:.3f}s")
        cap.release()

    except FailedToConnect as ex:
        logger.error(f"Failed to connect to server: {ex}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Terminating ...")
    except Exception as ex:
        traceback.print_exc()
        logger.error(f"Exception: {repr(ex)}")
        sys.exit(1)
    finally:
        if collision_warning_client is not None:
            collision_warning_client.stop()


if __name__ == "__main__":
    main()
