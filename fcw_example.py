import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from collision import *
from detection import *
from sort import Sort
from yolo_detector import YOLODetector

from PIL import Image, ImageDraw, ImageFont

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Configuration of the algorithm
config = Path("config/config.yaml")

# Camera settings - specific for the particular input
camera_config = Path("videos/video3.yaml")
video_file = Path("videos/video3.mp4").as_posix()

# video_file = "rtp://localhost:1234/"


font = ImageFont.truetype("data/UbuntuMono-B.ttf", 24, encoding="unic")

from more_itertools import windowed
from math import ceil

def segmentize(p:LineString, max_dist=10):
    pts = []
    for a, b in windowed(p.coords, n=2):
        seg = LineString([a,b])
        f = np.linspace(0, seg.length, ceil(seg.length/max_dist), endpoint=False)
        _pts = [seg.interpolate(x) for x in f]
        pts.extend(_pts)
    return LineString(pts)


def draw_horizon(d: ImageDraw.ImageDraw, cam:Camera, **kwargs):
    # h = segmentize(cam.horizon, max_dist=20)
    x = list(cam.horizon.coords)
    # n = x.shape[1]
    # x = np.vstack([x, np.ones((1,n))]).astype(np.float32)
    # x = (np.linalg.inv(cam.K_new) @ x)[:2].T
    # X = cv2.fisheye.distortPoints(x.reshape(1,-1,2), cam.K, cam.D)[0]
    d.line(x, **kwargs)


def draw_tracked_objects(d: ImageDraw.ImageDraw, tracked_objects):
    for tid, t in tracked_objects.items():
        x1, y1, x2, y2 = t.get_state()[0]
        color = (0, 255, 0, 64)
        d.rectangle((x1, y1, x2, y2), fill=color, outline=None, width=0.5)
        # label = f"track {tid}"
        # _, _, tw, th = font.getbbox(label, stroke_width=1)
        # tw, th = tw + 4, th + 4
        # d.rectangle((x1, y1 - th, x1 + tw, y1), fill=(0, 0, 0))
        # d.text((x1 + 3, y1 - th + 2), label, fill=(255, 255, 255), font=font, stroke_width=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.info("Starting Forward Collision Guard")

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

    
    # output = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, camera.rectified_size)


    # FCW Loop
    while True:
        ret, img = video.read()
        if not ret or img is None:
            logging.info("Video ended")
            break
        img_undistort = camera.rectify_image(img)
        # Detect object in image
        dets = detector.detect(img_undistort)
        # Get bounding boxes as numpy array
        dets = detections_to_numpy(dets)
        # Update state of image trackers
        tracker.update(dets)
        # Represent trackers as dict  tid -> KalmanBoxTracker
        tracked_objects = {
            t.id: t for t in tracker.trackers
            if t.hit_streak > tracker.min_hits and t.time_since_update < 1
        }
        # Get 3D locations of objects
        ref_pt = get_reference_points(tracked_objects, camera, is_rectified=True)
        # print(ref_pt)
        # Update state of objects in world
        guard.update(ref_pt)
        # Get list of current offenses
        dangerous_objects = guard.dangerous_objects()

        # Visualization
        base = Image.fromarray(img_undistort[..., ::-1], "RGB").convert("RGBA")
        objects_image = Image.new("RGBA", base.size)
        osd_image = Image.new("RGBA", base.size)

        osd_draw = ImageDraw.Draw(osd_image)
        draw_horizon(osd_draw, camera, fill=(255,255,0,128), width=1)

        objects_draw = ImageDraw.Draw(objects_image)
        draw_tracked_objects(objects_draw, tracked_objects)

        for tid, o in dangerous_objects.items():
            x1, y1, x2, y2 = tracked_objects[tid].get_state()[0]
            # objects_draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            objects_draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64))
            dist = Point(o.location).distance(guard.vehicle_zone)
            info = f"{dist:.1f} m"
            objects_draw.text((0.5 * (x1 + x2), 0.5 * (y1 + y2)), info, align="center", font=font,
                              stroke_fill=(255, 255, 255), stroke_width=1, fill=(0, 0, 0))

        display = Image.alpha_composite(objects_image, osd_image)
        out = Image.alpha_composite(base, display).convert("RGB")

        cv_image = np.array(out)[..., ::-1]
        cv2.imshow("FCW", cv_image)
        cv2.waitKey(1)

    #     output.write(cv_image)

    # output.release()
