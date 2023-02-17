import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
import cv2

@dataclass(eq=False, order=False)
class Camera:
    image_size: tuple
    K:np.ndarray
    D:np.ndarray
    R:np.ndarray
    T:np.ndarray

    @property
    def K_inv(self):
        return inv(self.K)
    
    @property
    def RT(self):
        return inv(self.T @ self.R)

    @staticmethod
    def from_dict(d):
        K = np.eye(4)
        K[:3,:3] = np.array(d["K"], "f")
        D = np.array(d["D"], "f")
        view_direction = d.get("view_direction", "x")
        R = np.eye(4)
        R[:3,:3] = estimate_R(d["K"], d["horizon"], view_direction)
        T = translation_matrix(d.get("location", [0,0,1]))
        # P = K @ np.linalg.inv(T @ R)
        return Camera(d["image_size"], K, D, R, T)


def estimate_R(K, h, view_direction="x"):
    """
    Estimate rotation matrix from horizon observation
    """
    assert view_direction in {"x","-x"}
    x1,y1,x2,y2  = h
    # Compose matrix with homogeneous points
    h = np.array([
            [x1,y1,1],  # Point in X direction
            [x2,y2,1]   # Point elsewhere on the horizon
        ]).T

    # Directions to the points
    H = (np.linalg.inv(K) @ h).T

    # Calc direction of the individual axes
    
    direct_sign = +1 if view_direction=="x" else -1
    direct = direct_sign * H[0]

    up = np.cross(H[1], H[0])
    up = -np.sign(up[1]) * up

    right = np.cross(up, direct)

    R  = np.array([direct, right, up])
    n = np.linalg.norm(R, axis=1, keepdims=True)

    return R / n


def translation_matrix(t):
    T = np.eye(4)
    T[:3,-1] = t
    return T


def project_screen_points_to_plane(x, K, RT, plane_normal):
    """
    Get a 3D world position of observed point
    """
    RT_inv = inv(RT)
    K_inv = inv(K)
    X = RT_inv @ K_inv @ x  # USe just R - dont need to use O and then X = S
    O = RT_inv @ np.atleast_2d([0,0,0,1]).T  # This is T vector
    # print(O)
    S = X - O
    n = np.atleast_2d([0,0,1])
    t = (plane_normal @ O[:3]) / (plane_normal @ S[:3])
    # print(t)
    return O - t * S


def object_world_space_coords(x, K, D, RT):
    """
    x: (N,2)

    Returns:
    X: (N,3)
    """
    n = x.shape[0]
    if n == 0:
        return np.empty((4,0))
    x = np.atleast_2d(cv2.fisheye.undistortPoints(x.reshape(1,-1,2), K[:3,:3], D, None, K[:3,:3]).squeeze())
    x = np.hstack([x, np.ones((x.shape[0],2))]).T
    # World points
    return project_screen_points_to_plane(
        x, K, RT, np.atleast_2d([0,0,1])
    )[:3].T