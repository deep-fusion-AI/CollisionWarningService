import cv2
import numpy as np
# Fix horrible naming of OpenCV functions
from cv2.fisheye import (
    estimateNewCameraMatrixForUndistortRectify as estimateCameraMatrix,
    initUndistortRectifyMap,
    undistortPoints,
    distortPoints,
)
from numpy.linalg import inv
from shapely.geometry import LineString, Point


def wpca(X, w):
    """ Weighted PCA """
    U, E, _ = np.linalg.svd(X.T @ np.diag(w) @ X)
    return U, E


def fit_line(x):
    n = x.shape[0]
    pts = np.vstack([x.T, np.ones((1, n))])
    pts = pts / np.linalg.norm(pts, axis=0, keepdims=True)
    v, w = wpca(pts.T, np.ones(n))
    p = v[:, np.argmin(w)]
    return p / p[2]  # Line params


def line_segment(l, x0, x1):
    left = np.array([-1, 0, x0])
    right = np.array([-1, 0, x1])

    hl = np.cross(l, left)
    hr = np.cross(l, right)

    x1, y1 = hl[:2] / hl[2]
    x2, y2 = hr[:2] / hr[2]

    return x1, y1, x2, y2


class Camera:
    def __init__(self, image_size, rectified_size, K, D, RT=None):
        self.horizon = None
        self.RT_inv = None
        self.image_size = image_size
        self.rectified_size = rectified_size
        self.K = K
        self.D = D
        self.RT = RT or np.eye(3, 4)

        # Rectification parameters
        self.K_new = estimateCameraMatrix(
            self.K, self.D, tuple(self.image_size), np.eye(3), new_size=tuple(self.rectified_size), fov_scale=1.2
        )
        self.maps = initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K_new, tuple(self.rectified_size), cv2.CV_32F)

        # view_direction = d.get("view_direction", "x")
        # R = np.eye(4)
        # R[:3,:3] = estimate_R(d["K"], d["horizon"], view_direction)
        # T = translation_matrix(d.get("location", [0,0,1]))
        # self.RT = inv(self.T @ self.R)

    def project_points(self, X, near: float = 0, to_rectified: bool = True):
        """
        X : (N,3) matrix
        """
        n = X.shape[0]
        X = np.vstack([X.T, np.ones((1, n))])
        K = self.K_new if to_rectified else self.K
        x = K @ self.RT @ X
        d = x[2]
        valid = d > near
        x = x[:2, valid] / d[valid]
        return x.T, d[valid]

    def rectify_image(self, image):
        map1, map2 = self.maps
        img = cv2.GaussianBlur(image, (3, 3), 0.5)
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def rectify_points(self, x):
        y = undistortPoints(x.reshape(1, -1, 2), self.K, self.D, P=self.K_new)
        return y[0]

    def unrectify_points(self, x):
        """
        x : (n,2) in K_new
        """
        n = x.shape[0]
        x = np.vstack([x.T, np.ones((1,n))]).astype(np.float32)
        x_norm = inv(self.K_new) @ x 
        y = distortPoints(x_norm.reshape(1,-1,2), self.K, self.D)
        return y[0]

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        image_size = d["image_size"]
        rectified_size = d["rectified_size"]
        # Measured intrinsic matrix
        K = np.array(d["K"], "f")
        # Measured distortion coefficients
        D = np.array(d["D"], "f")
        cam = Camera(image_size, rectified_size, K, D)

        w, h = image_size
        rw, rh = rectified_size

        # Estimate rotation matrix
        # Get points on horizon or simply one point in image center
        h_points = np.atleast_2d(d.get("horizon_points", [(w / 2, h / 2)])).astype(np.float32)
        # Get undistorted coords of the points
        h_points = cam.rectify_points(h_points)
        if h_points.shape[0] == 1:  # Single point case - add second point on the same line
            x, y = h_points[0]
            h_points = np.vstack([h_points, (x + 100, y)])
        # Line segment in undistorted image defining the horizon
        x1, y1, x2, y2 = line_segment(fit_line(h_points), 0, rw)
        # Represent horizon as LineString - convenient for geometric processing
        cam.horizon = LineString([(x1, y1), (x2, y2)])
        # Get location of first point on the horizon - the closest point on the line
        x1, y1 = cam.horizon.interpolate(cam.horizon.project(Point(h_points[0]))).coords[0]
        # Estimate the rotation matrix
        R = np.eye(4)
        R[:3, :3] = estimate_R(cam.K_new, (x1, y1, x2, y2), d.get("view_direction", "x"))
        # add translation
        T = translation_matrix(d.get("location", [0, 0, 1]))
        cam.RT = np.linalg.inv(T @ R)[:3]
        cam.RT_inv = (T @ R)[:3]

        return cam


def estimate_R(K, h, view_direction="x"):
    """
    Estimate rotation matrix from horizon observation
    """
    assert view_direction in {"x", "-x"}
    x1, y1, x2, y2 = h
    # Compose matrix with homogeneous points
    h = np.array(
        [
            [x1, y1, 1],  # Point in X direction
            [x2, y2, 1]  # Point elsewhere on the horizon
        ]
    ).T

    # Directions to the points
    H = (np.linalg.inv(K) @ h).T

    # Calc direction of the individual axes

    direct_sign = +1 if view_direction == "x" else -1
    direct = direct_sign * H[0]

    up = np.cross(H[1], H[0])
    up = -np.sign(up[1]) * up

    right = np.cross(up, direct)

    R = np.array([direct, right, up])
    n = np.linalg.norm(R, axis=1, keepdims=True)

    return R / n


def translation_matrix(t):
    T = np.eye(4)
    T[:3, -1] = t
    return T
