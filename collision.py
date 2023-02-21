import logging
from typing import Dict

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from shapely.geometry import LineString, Point, Polygon, box

from geometry import *


def F_matrix(dt):
    dt2 = 0.5*dt**2
    return np.array([[1,  dt, dt2,   0,   0,   0],
                     [0,   1,  dt,   0,   0,   0],
                     [0,   0,   1,   0,   0,   0],
                     [0,   0,   0,   1,  dt, dt2],
                     [0,   0,   0,   0,   1,  dt],
                     [0,   0,   0,   0,   0,   1]], dtype=np.float32)

def object_tracker(x_init, dt=1):
    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.F = F_matrix(dt)

    kf.H = np.array([[1,0,0,0,0,0],
                     [0,0,0,1,0,0]])    # Measurement function

    kf.P = np.diag([1,2,4,1,2,4]) * 0.5
    z_std = 2
    kf.R = np.diag([z_std**2, z_std**2]) # 1 standard  
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.1e-0**2, block_size=2) # process uncertainty
    kf._alpha_sq = 1
    x,y = x_init
    kf.x[0] = x
    kf.x[3] = y
    return kf


class PointWorldObject:
    """
    Simplest abstraction of world objects - just a location

    One can implement 
    """
    def __init__(self, xyz:np.ndarray, dt:float):
        self.kf = object_tracker(xyz[:2], dt=dt)
        self.xy = None
        self.vxvy = None

    def update(self, location=None):
        self.kf.predict()
        self.kf.update(location)
        self.xy = np.dot(self.kf.H, self.kf.x).T[0]
        self.vxvy = np.dot(np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]]), self.kf.x).T[0]

    @property
    def location(self):
        return self.xy

    @property
    def distance(self):
        if self.xy is None: return np.inf
        return np.linalg.norm(self.xy)

    @property
    def relative_speed(self):
        if self.vxvy is None: return 0
        return np.linalg.norm(self.vxvy)

    def future_path(self, length:float=1, dt:float=0.1):
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


def get_reference_points(trackers:dict, camera:Camera):
    """
    Convert 2D observation to 3D
    """
    if not trackers: return dict()

    # (xyxy) -> (rx,ry)
    R = np.array(
        [[0.5, 0, 0.5, 0],
        [  0, 0,   0, 1]],
    )

    bb = np.vstack([tracker.get_state() for tracker in trackers.values()]).T
    img_rp = R @ bb  # (N,2)
    world_rp = object_world_space_coords(img_rp, camera.K, camera.D, camera.RT)
    return dict(zip(trackers.keys(), world_rp))  # tid -> (x,y,z)


class ForwardCollisionGuard:
    def __init__(
            self,
            danger_zone:Polygon,
            vehcile_zone:Polygon,
            safety_radius:float = 25,
            prediction_length:float = 1,
            prediction_step:float = 0.1,
            dt:float = 1,
        ):
        self.dt = dt
        self.objects:Dict[int,PointWorldObject] = dict()
        self.danger_zone = danger_zone
        self.vehicle_zone = vehcile_zone
        self.safety_radius = safety_radius  # m
        self.prediction_length = prediction_length
        self.prediction_step = prediction_step

    @staticmethod
    def from_dict(d):
        zone = Polygon(d.get("danger_zone"))
        length, width = d.get("vehcile_length",4), d.get("vehicle_width",1.8)
        vehicle_zone = box(-length/2, -width/2, length/2, width/2).buffer(0.5, resolution=4)
        
        return ForwardCollisionGuard(
            danger_zone=zone,
            vehcile_zone=vehicle_zone,
            safety_radius=d.get("safety_radius", 30),
            prediction_length=d.get("prediction_length", 1),
            prediction_step=d.get("prediction_step", 0.1),
        )

    def update(self, ref_points:dict):
        """
        Update state of objects tracked in world space
        """
        # Sync world trackers with image tarckers
        for tid in list(self.objects.keys()):
            if tid not in ref_points:
                self.objects.pop(tid)
                logging.info(f"Tracking of {tid} lost")

        for tid in ref_points.keys():
            if tid not in self.objects:
                logging.info("Tracking object {tid}".format(tid=tid))
                self.objects[tid] = PointWorldObject(ref_points[tid], self.dt)
            else:
                self.objects[tid].update(ref_points[tid][:2])
    
    def dangerous_objects(self):
        """
        Check future paths of objects and filter dangerous ones
        """
        return {
            tid: obj for tid, obj in self.objects.items()
            if obj.distance < self.safety_radius and
               obj.future_path(self.prediction_length, self.prediction_step).intersects(self.danger_zone)
        }


###########        
        # return w
        

        # # Update trackers with new reference point
        # for _rp, _track, in zip(rp_world, tracks):

        #     track_id = int(_track[-1])
        #     if track_id not in world_trackers.keys():
        #         world_trackers[track_id] = object_tracker(_rp, dt = dt)

        #     kf = world_trackers[track_id]

        #     kf.update(_rp.copy())

        # # Predict future path of close objects
        # fp = dict()
        # for kf in world_trackers.values():
        #     X = predict_future_path(kf, length=1, dt=0.1)
        #     n = X.shape[1]
        #     X1 = np.vstack([
        #         X[[0,3],:],
        #         np.zeros((1,n)),  # z=0
        #         np.ones((1,n)),
        #     ])

        #     x = K @ RT @ X1
        #     d = x[2]
        #     x = x[:2,d>0] / x[2,d>0]
        #     for _x,_y in x.T:
        #         cv2.circle(img_und, (int(_x), int(_y)), 5, color=(0,128,255), thickness=-1)

        #     path = LineString(X1[:2].T)

        #     if path.intersects(safety_poly):
        #         cv2.circle(img_und, (50, 50), 20, color=(0,0,255), thickness=-1)


