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


def get_horizon_line(h, image_size):
    ...


def draw_horizon(d:ImageDraw.ImageDraw, horizon):
    d.line(horizon, (0,255,0), 2)
    ...


def draw_tracked_objects(d:ImageDraw.ImageDraw, tracked_objects):
    for tid,t in tracked_objects.items():
        x1,y1,x2,y2 = t.get_state()[0]
        color = (255,255,0,64)
        d.rectangle((x1,y1,x2,y2), fill=color, outline=None, width=0.5)
        label = f"track {tid}"
        tw, th = font.getsize(label, stroke_width=1)
        tw, th = tw+4, th+4
        d.rectangle((x1,y1-th,x1+tw,y1), fill=(0,0,0))
        d.text((x1+3,y1-th+2), label, fill=(255,255,255), font=font, stroke_width=1)


# def undistort(img, camera:Camera):
#     sz = camera.image_size
#     K1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (640, 360), np.eye(3), balance=0.5, new_size=(640,360))
#     print(K1)
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K1, (640,360), cv2.CV_32F)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     return undistorted_img


# class Undistorter:
#     def __init__(self, camera:Camera, new_size=None, balance:float=0.5):
#         self.cam = camera
#         self.new_size = new_size or camera.image_size
#         R = np.eye(3)
#         K = camera.K[:3,:3]
#         D = camera.D
#         self.K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
#             K, D, camera.image_size, R, balance=balance, new_size=self.new_size)
#         self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
#             K, D, R, self.K, self.new_size, cv2.CV_32F)
        
#     def undistort_image(self, img):
#         return cv2.remap(
#             img, self.map1, self.map2,
#             interpolation=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_CONSTANT)
        

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
    
    # output = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

    # Load camera calibration
    logging.info("Loading camera configuration {cfg}".format(cfg=camera_config))
    camera_dict = yaml.safe_load(camera_config.open())
    camera = Camera.from_dict(camera_dict)

    # Open video
    logging.info("Openning video {vid}".format(vid=video_file))
    video = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = height, width
    logging.info("Video {W}x{H}, {fps} FPS".format(W=width, H=height, fps=fps))

    guard.dt = 1/fps  # Finish setup if the guard

    # FCW Loop
    while True:
        ret, img = video.read()
        if not ret or img is None:
            logging.info("Video ended")
            break

        # Detect object in image
        dets = detector.detect(cv2.GaussianBlur(img, (5,5), 1))
        # Get bounding boxes as numpy array
        dets = detections_to_numpy(dets)
        # Update state of image trackers
        tracker.update(dets)
        # Represent trackers as dict  tid -> KalmanBoxTracker
        tracked_objects = {
            t.id:t for t in tracker.trackers
            if t.hit_streak>tracker.min_hits and t.time_since_update<1
        }
        # Get 3D locations of objects
        ref_pt = get_reference_points(tracked_objects, camera)
        # Update state of objects in world
        guard.update(ref_pt)
        # Get list of current offenses
        offending_objects = guard.ofsenses()

        # Vizialization
        base = Image.fromarray(img[...,::-1], "RGB").convert("RGBA")
        objects_image = Image.new("RGBA", base.size, (255, 255, 255, 0))
        osd_image = Image.new("RGBA", base.size, (255, 255, 255, 0))

        osd_draw = ImageDraw.Draw(osd_image)
        draw_horizon(osd_draw, camera_dict["horizon"])


        objects_draw = ImageDraw.Draw(objects_image)
        draw_tracked_objects(objects_draw, tracked_objects)

        for tid, (o,_,_) in offending_objects.items():
            x1,y1,x2,y2 = tracked_objects[tid].get_state()[0]
            objects_draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)

            info=f"{o.distance():.0f} m"
            objects_draw.text((0.5*(x1+x2), 0.5*(y1+y2)), info, align="center", font=font, stroke_fill=(255,255,255), stroke_width=2, fill=(0,0,0))
        
        display = Image.alpha_composite(objects_image, osd_image)
        out = Image.alpha_composite(base, display).convert("RGB")

        cv_image = np.array(out)[...,::-1]
        cv2.imshow("FCW", cv_image)
        cv2.waitKey(1)
        # output.write(cv_image)

    # output.release()