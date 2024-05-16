"""
Early Collision Warning system
"""
import json
from argparse import ArgumentParser, FileType
import statistics
from datetime import datetime
import cv2
import yaml
import time
import csv
import logging
import sys

from era_5g_interface.measuring import Measuring

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FCW example")

from fcw_core_utils.collision import get_reference_points, ForwardCollisionGuard
from fcw_core.detection import detections_to_numpy
from fcw_core.sort import Sort
from fcw_core.yolo_detector import YOLODetector

from fcw_core.vizualization import *
from fcw_core_utils.rate_timer import RateTimer

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

start_timestamp = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", type=FileType("r"), required=True, help="Collision warning config")
    parser.add_argument("--camera", type=FileType("r"), required=True, help="Camera settings")
    parser.add_argument("-o", "--output", type=str, help="Output video")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("-t", "--play_time", type=int, help="Video play time in seconds", default=60)
    parser.add_argument("--fps", type=int, help="Video FPS", default=None)
    parser.add_argument("source_video", type=str, help="Video stream (file or url)")

    return parser.parse_args()


def main(args=None):
    args = parse_arguments()
    logger.info("Starting Forward Collision Guard")

    logger.info("Loading configuration file {cfg}".format(cfg=args.config.name))
    config_dict = yaml.safe_load(args.config)

    # Open video
    logger.info("Opening video {vid}".format(vid=args.source_video))
    video = cv2.VideoCapture(args.source_video)
    if not video.isOpened():
        raise Exception("Cannot open video file")
    if not args.fps:
        fps = video.get(cv2.CAP_PROP_FPS)
    else:
        fps = args.fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = height, width
    logger.info("Video {W}x{H}, {fps} FPS".format(W=width, H=height, fps=fps))

    # Init object detector
    detector = YOLODetector.from_dict(config_dict.get("detector", {}))

    # Init image tracker
    logger.info("Initializing image tracker")
    tracker = Sort.from_dict(config_dict.get("tracker", {}))
    tracker.dt = 1 / fps

    # Init collision guard
    logger.info("Initializing Forward warning")
    guard = ForwardCollisionGuard.from_dict(config_dict.get("fcw", {}))
    guard.dt = 1 / fps  # Finish setup if the guard

    # Load camera calibration
    logger.info("Loading camera configuration {cfg}".format(cfg=args.camera.name))
    camera_dict = yaml.safe_load(args.camera)
    camera = Camera.from_dict(camera_dict)

    render_output = args.viz or args.output is not None
    if render_output:
        logger.warning("RENDERING OUTPUT - LOWER PERFOMANCE")

    if args.output is not None:
        output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"MP4V"), fps, camera.image_size)

    if args.viz:
        try:
            cv2.namedWindow("FCW")
        except Exception as ex:
            logger.debug(repr(ex))

    if render_output:
        # Prepare static stuff for visualization
        logo = cog_logo((64, 64))
        coord_sys = draw_world_coordinate_system(camera.rectified_size, camera)
        coord_sys.putalpha(64)
        danger_zone = draw_danger_zone(camera.rectified_size, camera, guard.danger_zone)
        horizon = draw_horizon(camera.rectified_size, camera, width=1, fill=(255, 255, 0, 64))
        marker, marker_anchor = vehicle_marker_image(scale=3)

    delays = []
    measuring_items = {
        "key_timestamp": 0,
        "final_timestamp": 0,
        "worker_recv_timestamp": 0,
        "worker_before_process_timestamp": 0,
        "worker_after_process_timestamp": 0,
        "worker_send_timestamp": 0,
    }
    prefix = f"client-final"
    measuring = Measuring(measuring_items, enabled=True, filename_prefix=prefix)

    rate_timer = RateTimer(rate=fps, iteration_miss_warning=True)

    # FCW Loop
    start_time = time.time_ns()
    while time.time_ns() - start_time < args.play_time * 1.0e+9:
        ret, img = video.read()
        if not ret or img is None:
            logger.info("Video ended")
            break
        key_timestamp = time.perf_counter_ns()

        img_undistorted = camera.rectify_image(img)
        time0 = time.perf_counter_ns()
        measuring.log_measuring(key_timestamp, "worker_recv_timestamp", time0)
        measuring.log_measuring(key_timestamp, "worker_before_process_timestamp", time0)
        # Detect object in image
        detections = detector.detect(img_undistorted)
        # Get bounding boxes as numpy array
        detections = detections_to_numpy(detections)
        # Update state of image trackers
        tracker.update(detections)
        # Represent trackers as dict  tid -> KalmanBoxTracker
        tracked_objects = {
            t.id: t for t in tracker.trackers
            if t.hit_streak > tracker.min_hits and t.time_since_update < 1 and t.age > 3
        }

        # Get 3D locations of objects
        ref_pt = get_reference_points(tracked_objects, camera, is_rectified=True)
        # Update state of objects in world
        guard.update(ref_pt)
        # Get list of current offenses
        dangerous_objects = guard.dangerous_objects()

        time1 = time.perf_counter_ns()
        measuring.log_measuring(key_timestamp, "worker_after_process_timestamp", time1)
        measuring.log_measuring(key_timestamp, "worker_send_timestamp", time1)
        logger.info(f"Delay: {(time1 - time0) * 1.0e-9:.3f}s")
        delays.append((time1 - time0))
        measuring.log_measuring(key_timestamp, "final_timestamp", time1)
        measuring.store_measuring(key_timestamp)

        if render_output:
            # Visualization
            base_undistorted = Image.fromarray(img_undistorted[..., ::-1], "RGB").convert("L").convert("RGBA")
            # Base layer is the camera image
            base = Image.fromarray(img[..., ::-1], "RGB").convert("RGBA")
            # Layers showing various information
            sz = base_undistorted.size
            layers = [
                (coord_sys, None),
                (danger_zone, None),
                (horizon, None),
                (draw_image_trackers(sz, tracker.trackers), None),
                (draw_world_objects(sz, camera, guard.objects.values()), None),
            ]

            # Compose layers together
            compose_layers(base_undistorted, *layers)
            O = list(guard.label_objects(include_distant=False))
            w, h = base.size
            w1, h1 = base_undistorted.size
            compose_layers(
                base,  # Original image
                (tracking_info((w, 16), O), (0, 0)),
                (mark_vehicles(camera.image_size, guard.objects.values(), camera, marker, marker_anchor), None),
                (logo, (8, 16 + 8)),
                (base_undistorted, (8, h - h1 - 8)),  # Pic with rectified image and vizualized trackers
            )
            # Convert to OpenCV for display
            cv_image = np.array(base.convert("RGB"))[..., ::-1]

        if args.viz:
            try:
                # Display the image
                cv2.imshow("FCW", cv_image)
                cv2.waitKey(1)
            except Exception as ex:
                logger.debug(repr(ex))

        if args.output is not None:
            output.write(cv_image)

        rate_timer.sleep()  # sleep until next frame should be sent (with given fps)

    if args.output is not None:
        output.release()

    logger.info(f"-----")
    end_time = time.time_ns()
    logger.info(f"Total streaming time: {(end_time - start_time) * 1.0e-9:.3f}s")
    logger.info(f"Delay median: {statistics.median(delays) * 1.0e-9:.3f}s")

    try:
        cv2.destroyAllWindows()
    except Exception as ex:
        logger.debug(repr(ex))


if __name__ == "__main__":
    main()
