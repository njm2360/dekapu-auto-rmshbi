import cv2
import numpy as np
from typing import Final, Optional

from app.point import Point


class MotionDetector:

    THRESHOLD_VALUE: Final[int] = 30
    MIN_CONTOUR_AREA: Final[int] = 200
    DILATE_ITERATIONS: Final[int] = 2

    def __init__(self, mask: Optional[np.ndarray] = None):
        self.mask = mask

    def detect(self, prev_img: np.ndarray, curr_img: np.ndarray) -> list[np.ndarray]:
        diff = cv2.absdiff(prev_img, curr_img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=self.DILATE_ITERATIONS)

        if self.mask is not None:
            dilated = cv2.bitwise_and(dilated, self.mask)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        filtered = [c for c in contours if cv2.contourArea(c) >= self.MIN_CONTOUR_AREA]

        if not filtered:
            return []

        filtered.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        return filtered

    def debug(self, contours, image):
        output = image.copy()

        for c in contours:
            area = cv2.contourArea(c)
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(c)
            text_pos = (x, y - 5 if y - 5 > 0 else y + 15)

            cv2.putText(
                output,
                f"{int(area)}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite("output.png", output)
