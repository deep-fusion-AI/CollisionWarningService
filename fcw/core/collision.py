from typing import Dict
from dataclasses import dataclass
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from shapely.geometry import LineString, Point, Polygon, box
import logging
logger = logging.getLogger(__name__)

from fcw.core.geometry import *


def F_matrix(dt):
    dt2 = 0.5 * dt ** 2
    return np.array(
        [[1, dt, dt2, 0, 0, 0],
         [0, 1, dt, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, dt, dt2],
         [0, 0, 0, 0, 1, dt],
         [0, 0, 0, 0, 0, 1]], dtype=np.float32
    )


def object_tracker(x_init, dt: float = 1):
    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.F = F_matrix(dt)

    kf.H = np.array(
        [[1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0]]
    )  # Measurement function

    kf.P = np.diag([1, 2, 400, 1, 2, 400]) * 10
    z_std = 2
    kf.R = np.diag([z_std ** 2, z_std ** 2])  # 1 standard
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.5e-1 ** 2, block_size=2)  # process uncertainty
    kf._alpha_sq = 1
    x, y = x_init
    kf.x[0] = x
    kf.x[3] = y
    return kf


def covariance(xy: np.ndarray, sigma: float = 0.1, scale: float = 0.1):
    d = np.linalg.norm(xy)
    x, y = xy
    th = np.arctan2(y, x)
    c, s = np.cos(th), np.sin(th)
    M = np.array([[c, -s], [s, c]])
    return M @ (np.diag([1, scale]) * sigma * d)


class PointWorldObject:
    """
    Simplest abstraction of world objects - just a location

    One can implement 
    """

    def __init__(self, xyz: np.ndarray, dt: float):
        self.kf = object_tracker(xyz[:2], dt=dt)
        self.xy = None
        self.vxvy = None

    def update(self, location=None):
        self.kf.predict()
        self.kf.update(location, R=covariance(location, sigma=0.1, scale=0.1))
        self.xy = np.dot(self.kf.H, self.kf.x).T[0]
        self.vxvy = np.dot(np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]), self.kf.x).T[0]

    @property
    def location(self):
        return self.xy

    @property
    def distance(self):
        if self.xy is None:
            return np.inf
        else:
            return np.linalg.norm(self.xy)

    @property
    def relative_speed(self):
        if self.vxvy is None: return 0
        return np.linalg.norm(self.vxvy)

    def future_path(self, length: float = 1, dt: float = 0.1):
        x = self.kf.x
        F = F_matrix(dt)
        t = 0
        X = [x]
        while t < length:
            x = np.dot(F, x)
            t += dt
            X.append(x)
        xy = self.kf.H @ np.hstack(X)  # (2, N)
        return LineString(xy.T)


def get_reference_points(trackers: dict, camera: Camera, *, is_rectified: bool):
    """
    Convert 2D observation to 3D
    """
    if not trackers: return dict()

    # (xyxy) -> (rx,ry)
    R = np.array(
        [[0.5, 0, 0.5, 0],
         [0, 0, 0, 1]],
    )
    # image space bounding boxes of objects
    bb = np.vstack([tracker.get_state() for tracker in trackers.values()]).T
    img_rp = R @ bb  # (2,N) 2D ref points in distorted image

    if not is_rectified:
        # If trackers are used on non-rectified image
        img_rp = camera.rectify_points(img_rp.T).T

    # points are in cam.K_new camera - normalize and convert to (3,N)
    n = img_rp.shape[1]
    img_rp_h = np.vstack([img_rp, np.ones((1, n))])
    norm_rp = inv(camera.K_new) @ img_rp_h  # (3,N) normalized xyz in camera space

    norm_rp = np.vstack([norm_rp, np.ones((1, n))])  # (4,N) homogeneous in 3D

    X = camera.RT_inv @ norm_rp  # (3,N)
    O = camera.RT_inv[:, -1].reshape(3, -1)
    S = X - O
    plane_normal = np.atleast_2d([0, 0, 1])
    t = (plane_normal @ O[:3]) / (plane_normal @ S[:3])
    world_rp = O - t * S

    return dict(zip(trackers.keys(), world_rp.T))  # tid -> (x,y,z)


class ForwardCollisionGuard:
    def __init__(
        self,
        danger_zone: Polygon,
        vehicle_zone: Polygon,
        safety_radius: float = 25,
        prediction_length: float = 1,
        prediction_step: float = 0.1,
        dt: float = 1,
    ):
        self.dt = dt
        self.objects: Dict[int, PointWorldObject] = dict()
        self.danger_zone = danger_zone
        self.vehicle_zone = vehicle_zone
        self.safety_radius = safety_radius  # m
        self.prediction_length = prediction_length
        self.prediction_step = prediction_step

    @staticmethod
    def from_dict(d):
        zone = Polygon(d.get("danger_zone"))
        length, width = d.get("vehicle_length", 4), d.get("vehicle_width", 1.8)
        vehicle_zone = box(-length / 2, -width / 2, length / 2, width / 2).buffer(0.5, resolution=4)

        return ForwardCollisionGuard(
            danger_zone=zone,
            vehicle_zone=vehicle_zone,
            safety_radius=d.get("safety_radius", 30),
            prediction_length=d.get("prediction_length", 1),
            prediction_step=d.get("prediction_step", 0.1),
        )

    def update(self, ref_points: dict):
        """
        Update state of objects tracked in world space
        """
        # Sync world trackers with image trackers
        for tid in list(self.objects.keys()):
            if tid not in ref_points:
                self.objects.pop(tid)
                logger.info(f"Tracking of {tid} lost")

        for tid in ref_points.keys():
            if tid not in self.objects:
                logger.info("Tracking object {tid}".format(tid=tid))
                self.objects[tid] = PointWorldObject(ref_points[tid], self.dt)
            else:
                self.objects[tid].update(ref_points[tid][:2])

    def dangerous_objects(self):
        """
        Check future paths of objects and filter dangerous ones
        """
        return {
            tid: obj for tid, obj in self.objects.items()
            if obj.distance < self.safety_radius and obj.future_path(
                self.prediction_length, self.prediction_step
            ).intersects(self.danger_zone)
        }

    def label_objects(
        self,
        include_distant: bool = False,
    ):
        """
        Check future paths of objects and filter dangerous ones
        """
        for tid, obj in self.objects.items():
            if obj.xy is None: continue

            loc = Point(obj.location)
            dist = loc.distance(self.vehicle_zone)

            if dist > self.safety_radius and not include_distant:
                continue

            path = obj.future_path(self.prediction_length, self.prediction_step)
            collision_point_distance = intersection_point(path, self.vehicle_zone.boundary)
            if collision_point_distance is not None:
                ttc = (collision_point_distance / path.length) * self.prediction_length
            else:
                ttc = None

            yield ObjectStatus(
                distance=dist,
                location=loc,
                path=path,
                is_in_danger_zone=self.danger_zone.contains(loc),
                crosses_danger_zone=path.crosses(self.danger_zone),
                time_to_collision=ttc,
            )


@dataclass
class ObjectStatus:
    # Distance from the reference point to vehicle zone
    distance: float
    # Location relative to vehicle reference point
    location: Point
    path: LineString
    # Flag indicating object in danger zone
    is_in_danger_zone: bool
    # Flag indicating if object path crosses danger zone
    crosses_danger_zone: bool
    # Time for reference point to reach vehicle zone. If None, does not reach vehicle
    time_to_collision: float

    @property
    def is_colliding(self):
        return self.time_to_collision is not None

    @property
    def is_dangerous(self):
        """
        Dangerous objects
        """
        danger = self.is_in_danger_zone
        collision = self.time_to_collision is not None and self.time_to_collision < 1
        return danger or collision


from more_itertools import pairwise


def intersection_point(ls: LineString, p: LineString):
    coords = ls.coords
    d = 0
    for a, b in pairwise(coords):
        l = LineString([a, b])
        pt = l.intersection(p)
        if not pt.is_empty:
            return Point(a).distance(pt) + d
        d += l.length
