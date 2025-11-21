import cv2
import random
from typing import Final
from enum import IntEnum
from numpy import ndarray

from app.point import Point


class ContourClickMode(IntEnum):
    CENTER = 0
    RANDOM = 1


class ClickPointExtractor:

    MIN_CLICK_DISTANCE: Final[int] = 20
    MAX_CLICKS_PER_LOOP: Final[int] = 10
    RANDOM_CLICKS_PER_CONTOUR: Final[int] = 3
    ENABLE_AVOID_CLOSE_CLICK: Final[bool] = True

    def __init__(
        self,
        mode: ContourClickMode = ContourClickMode.RANDOM,
    ):
        self.mode = mode

    def extract(self, contours: list[ndarray]) -> list[Point]:
        points: list[Point] = []

        for contour in contours:
            if len(points) >= self.MAX_CLICKS_PER_LOOP:
                break

            match self.mode:
                case ContourClickMode.RANDOM:
                    candidates = (
                        self._random_point(contour)
                        for _ in range(self.RANDOM_CLICKS_PER_CONTOUR)
                    )
                case ContourClickMode.CENTER:
                    candidates = (self._center_point(contour),)

            for point in candidates:
                if point is None:
                    continue
                if self.ENABLE_AVOID_CLOSE_CLICK:
                    if not self._far_enough(point, points):
                        continue
                points.append(point)

        return points

    def _random_point(self, contour: ndarray):
        bx, by, bw, bh = cv2.boundingRect(contour)
        for _ in range(5):
            x = random.randint(bx, bx + bw)
            y = random.randint(by, by + bh)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return Point(x, y)
        return None

    def _center_point(self, contour: ndarray):
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        return Point(x, y)

    def _far_enough(self, point: Point, points: list[Point]):
        return all(point.distance_to(p) >= self.MIN_CLICK_DISTANCE for p in points)
