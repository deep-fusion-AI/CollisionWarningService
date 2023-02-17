from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import Point, Polygon


@dataclass(eq=False, frozen=True, order=False)
class ObjectObservation:
    geometry: Polygon
    score: float
    label: int

    def bounds(self):
        return self.geometry.bounds
    
    def numpy(self):
        return np.atleast_2d(self.bounds() + (self.score,))
    
    def reference_point(self):
        """Center of bottom edge"""
        x1, _, x2, y2 = self.bounds()
        return Point(0.5 * (x1+x2), y2)
    
    def is_in_frame(self, shape, margin=5):
        h,w = shape
        x1, y1, x2, y2 = self.bounds()
        return (x1 > margin) and (x2 < (w-margin)) and (y1 > margin) and (y2 < (h-margin))


def detections_to_numpy(dets: Iterable[ObjectObservation]) -> np.ndarray:
    dets = [d.numpy() for d in dets]
    dets.append(np.empty((0,5)))  # vstack does not accept empty list - we add empty array so it does not fail
    return np.vstack(dets)
