from __future__ import annotations

import traceback
from argparse import ArgumentParser
from pathlib import Path
import cv2
import sys
import logging
from typing import Any, Dict

# FCW and 5G-ERA stuff.
from fcw_client.client_common import CollisionWarningClient
from era_5g_client.exceptions import FailedToConnect
from era_5g_interface.utils.rate_timer import RateTimer

# Set logging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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


def results_callback(results: Dict[str, Any]) -> None:
    """Callback which process the results from the FCW service.

    Args:
        results (Dict[str, Any]): The results in JSON format.
    """

    # Process detections.
    if "dangerous_detections" in results:
        for tracked_id, detection in results["dangerous_detections"].items():
            score = float(detection["dangerous_distance"])
            if score > 0:
                logger.info(f"Dangerous distance {score:.2f}m to the object with id {tracked_id}")


def main() -> None:
    """Example of simple FCW client."""

    # Parse arguments.
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Collision warning config", default=CONFIG_FILE)
    parser.add_argument("--camera", type=Path, help="Camera settings", default=CAMERA_CONFIG_FILE)
    parser.add_argument("source_video", type=str, help="Video stream (file or url)", nargs="?", default=TEST_VIDEO_FILE)
    args = parser.parse_args()

    collision_warning_client = None

    try:
        # Create a video capture to pass images to the 5G-ERA Network Application.
        logger.info(f"Opening video capture {args.source_video}")
        cap = cv2.VideoCapture(args.source_video)
        if not cap.isOpened():
            raise Exception("Cannot open video capture")

        # FPS can be wrong, it should be checked.
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create collision warning client with given parameters,
        # width and height for client registration is loaded from camera config.
        collision_warning_client = CollisionWarningClient(
            config=args.config, camera_config=args.camera, fps=fps, results_callback=results_callback
        )

        # Rate timer for control the speed of a loop (fps).
        rate_timer = RateTimer(rate=fps)

        # Main processing loop.
        logger.info("Start sending images...")
        while True:
            # Read single frame from a stream.
            ret, frame = cap.read()
            if not ret:
                break
            # Send to FCW service for detection/processing.
            collision_warning_client.send_image(frame)
            # Sleep until next frame should be sent (with given fps).
            rate_timer.sleep()
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
