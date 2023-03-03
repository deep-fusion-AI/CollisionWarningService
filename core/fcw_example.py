"""
Early Collision Warning system


"""

import logging
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

# Configuration of the algorithm
config = Path("../config/config.yaml")

# Camera settings - specific for the particular input
camera_config = Path("../videos/video3.yaml")
video_file = Path("../videos/video3.mp4").as_posix()

camera_config = Path("../__videos/video5.yaml")
video_file = Path("../__videos/video5.mp4").as_posix()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Forward Collision Guard")

    logging.info("Loading configuration file {cfg}".format(cfg=config))
    config_dict = yaml.safe_load(config.open())

    # Init object detector
    detector = YOLODetector.from_dict(config_dict.get("detector", {}))

    # Init image tracker
    logging.info("Initializing image tracker")
    tracker = Sort.from_dict(config_dict.get("tracker", {}))

    # Init collision guard
    logging.info("Initializing Forward warning")
    guard = ForwardCollisionGuard.from_dict(config_dict.get("fcw", {}))

    # Load camera calibration
    logging.info("Loading camera configuration {cfg}".format(cfg=camera_config))
    camera_dict = yaml.safe_load(camera_config.open())
    camera = Camera.from_dict(camera_dict)

    # Open video
    logging.info("Opening video {vid}".format(vid=video_file))
    video = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = height, width
    logging.info("Video {W}x{H}, {fps} FPS".format(W=width, H=height, fps=fps))

    guard.dt = 1 / fps  # Finish setup if the guard

    output = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, camera.image_size)

    cv2.namedWindow("FCW")

    logo = cog_logo((80, 80))

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

        # Vizualization
        # if args.show:
        base_undistorted = Image.fromarray(img_undistorted[..., ::-1], "RGB").convert("L").convert("RGBA")
        # Base layer is the camera image
        base = Image.fromarray(img[..., ::-1], "RGB").convert("RGBA")
        # Layers showing various information
        sz = base_undistorted.size
        layers = [
            (draw_danger_zone(sz, camera, guard.danger_zone), None),
            (draw_horizon(sz, camera, width=1, fill=(255,255,0,64)), None),
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
            (base_undistorted, (8, h-h1-8)),  # Pic with rectified image and vizualized trackers
            (tracking_info((w,16), O), (0,0)),
            (logo, (8,16+8))
        )
        # Convert to OpenCV for display
        cv_image = np.array(base.convert("RGB"))[...,::-1]
        # Display the image
        cv2.imshow("FCW", cv_image)
        cv2.waitKey(1)

        output.write(cv_image)

    output.release()

    cv2.destroyAllWindows()