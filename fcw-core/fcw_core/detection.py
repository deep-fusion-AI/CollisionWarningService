from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon


@dataclass(eq=False, frozen=True, order=False)
class ObjectObservation:
    geometry: Polygon
    score: float
    label: int

    def bounds(self):
        return self.geometry.bounds

    def numpy(self):
        return np.atleast_2d(self.bounds() + (self.score,) + (self.label,))

    def is_in_frame(self, shape, margin=5):
        h, w = shape
        x1, y1, x2, y2 = self.bounds()
        return (x1 > margin) and (x2 < (w - margin)) and (y1 > margin) and (y2 < (h - margin))


def detections_to_numpy(detections: Iterable[ObjectObservation]) -> np.ndarray:
    detections = [d.numpy() for d in detections]
    detections.append(np.empty((0, 6)))  # vstack does not accept empty list - we add empty array, so it does not fail
    return np.vstack(detections)
