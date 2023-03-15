"""
Early Collision Warning system
"""

import logging
from argparse import ArgumentParser, FileType
from pathlib import Path

import cv2
import numpy as np
import yaml
from collision import ForwardCollisionGuard, get_reference_points
from detection import detections_to_numpy
from geometry import Camera
from PIL import Image
from sort import Sort
from vizualization import *
from yolo_detector import YOLODetector

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", type=FileType("r"), required=True, help="Collision warning config")
    parser.add_argument("--camera", type=FileType("r"), required=True, help="Camera settings")
    parser.add_argument("-o", "--output", type=str, help="Output video")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("source_video", type=str, help="Video stream (file or url)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Forward Collision Guard")

    logging.info("Loading configuration file {cfg}".format(cfg=args.config.name))
    config_dict = yaml.safe_load(args.config)

     # Open video
    logging.info("Opening video {vid}".format(vid=args.source_video))
    video = cv2.VideoCapture(args.source_video, cv2.CAP_FFMPEG)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = height, width
    logging.info("Video {W}x{H}, {fps} FPS".format(W=width, H=height, fps=fps))

    # Init object detector
    detector = YOLODetector.from_dict(config_dict.get("detector", {}))

    # Init image tracker
    logging.info("Initializing image tracker")
    tracker = Sort.from_dict(config_dict.get("tracker", {}))
    tracker.dt = 1 / fps

    # Init collision guard
    logging.info("Initializing Forward warning")
    guard = ForwardCollisionGuard.from_dict(config_dict.get("fcw", {}))
    guard.dt = 1 / fps  # Finish setup if the guard

    # Load camera calibration
    logging.info("Loading camera configuration {cfg}".format(cfg=args.camera.name))
    camera_dict = yaml.safe_load(args.camera)
    camera = Camera.from_dict(camera_dict)

   


    render_output = args.viz or args.output is not None
    if render_output:
        logging.warning("RENDERING OUTPUT - LOWER PERFOMANCE")

    if args.output is not None:
        output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"MP4V"), fps, camera.image_size)

    if args.viz:
        cv2.namedWindow("FCW")

    if render_output:
        # Prepare static stuff for vizualization
        logo = cog_logo((64, 64))
        coord_sys = draw_world_coordinate_system(camera.rectified_size, camera)
        coord_sys.putalpha(64)
        danger_zone = draw_danger_zone(camera.rectified_size, camera, guard.danger_zone)
        horizon = draw_horizon(camera.rectified_size, camera, width=1, fill=(255,255,0,64))
        marker, marker_anchor = vehicle_marker_image(scale=3)

    # FCW Loop
    while True:
        ret, img = video.read()
        if not ret or img is None:
            logging.info("Video ended")
            break
        img_undistorted = camera.rectify_image(img)
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

        if render_output:
            # Vizualization
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
            O = list(guard.label_objects(include_distant=True))
            w,h  = base.size
            w1,h1 = base_undistorted.size
            compose_layers(
                base,   # Original image
                (tracking_info((w,16), O), (0,0)),
                (mark_vehicles(camera.image_size, guard.objects.values(), camera, marker, marker_anchor), None),
                (logo, (8,16+8)),
                (base_undistorted, (8, h-h1-8)),  # Pic with rectified image and vizualized trackers
            )
            # Convert to OpenCV for display
            cv_image = np.array(base.convert("RGB"))[...,::-1]
        
        if args.viz:
            # Display the image
            cv2.imshow("FCW", cv_image)
            cv2.waitKey(1)

        if args.output is not None:
            output.write(cv_image)

    if args.output is not None:
        output.release()

    cv2.destroyAllWindows()
