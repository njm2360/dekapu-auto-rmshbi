import cv2
import logging
import random
import asyncio
import keyboard
import pyautogui
import pygetwindow
import pydirectinput
import numpy as np

from enum import IntEnum
from dataclasses import dataclass
from typing import Final, Optional
from pygetwindow._pygetwindow_win import Win32Window


logging.basicConfig(level=logging.INFO)


@dataclass
class Point:
    x: int
    y: int


class ContourClickMode(IntEnum):
    CENTER = 0
    RANDOM = 1


class Image:
    THRESHOLD_VALUE: Final[int] = 30
    LOOP_WAIT_TIME: Final[int] = 5
    MIN_CONTOUR_AREA: Final[int] = 100
    MIN_CLICK_DISTANCE: Final[int] = 20
    MAX_CLICKS_PER_LOOP: Final[int] = 10

    TERMINATE_KEY: Final[str] = "esc"
    UPDATE_WINDOW_KEY: Final[str] = "f6"
    UPDATE_ORIGIN_KEY: Final[str] = "f7"
    RESET_KEY: Final[str] = "f8"

    def __init__(self, contour_click_mode: ContourClickMode = ContourClickMode.CENTER):
        self._task: asyncio.Task = None

        self._window: Optional[Win32Window] = None
        self._origin: Optional[Point] = None

        width, height = pyautogui.size()
        logging.info(f"Screen size : {width}x{height}")

        self._screen_width = width
        self._screen_height = height

        self._contour_click_mode = contour_click_mode

    async def keyboard_monitor_task(self):
        while True:
            if keyboard.is_pressed(self.TERMINATE_KEY):
                logging.info("Terminate key is pressed. Stopping...")
                if self._task:
                    self._task.cancel()
                break

            if keyboard.is_pressed(self.UPDATE_WINDOW_KEY):
                logging.info("Update target window")
                self.update_window()

            if keyboard.is_pressed(self.UPDATE_ORIGIN_KEY):
                logging.info("Update mouse origin")
                self.update_origin()

            if keyboard.is_pressed(self.RESET_KEY):
                logging.info("Reset window and origin")
                self._origin = None
                self._window = None

            await asyncio.sleep(0.1)

    def update_window(self) -> Win32Window:
        window = pygetwindow.getActiveWindow()
        if not window:
            logging.warning("Active window not found.")

        self._window = window

        logging.info(f"Window : {window}")

    def update_origin(self):
        p = pyautogui.position()
        self._origin = Point(p.x, p.y)
        logging.info(f"Origin : {self._origin.x} {self._origin.y}")

    async def main_loop(self):
        try:
            while True:
                await self.loop_step()
        except asyncio.CancelledError:
            pass
        finally:
            pydirectinput.keyUp("tab")

    async def loop_step(self):
        if not self.is_ready():
            return await self.sleep()

        prev_img, curr_img = await self.capture_window()
        if prev_img is None or curr_img is None:
            return await self.sleep()

        contours = self.get_contours(prev_img, curr_img)
        if not contours:
            return await self.sleep()

        points = self.extract_click_points(contours)
        if not points:
            return await self.sleep()

        result = await self.process_clicks(points)

        self.draw_debug(contours, points, result, curr_img)

        return await self.sleep()

    async def capture_window(self):
        try:
            img1 = pyautogui.screenshot(
                region=[
                    self._window.left,
                    self._window.top,
                    self._window.width,
                    self._window.height,
                ]
            )
            await asyncio.sleep(0.1)

            img2 = pyautogui.screenshot(
                region=[
                    self._window.left,
                    self._window.top,
                    self._window.width,
                    self._window.height,
                ]
            )
        except Exception as e:
            logging.warning(f"Screenshot failed: {e}")
            return None, None

        m1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        m2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

        return m1, m2

    def get_contours(self, prev_img, curr_img):
        diff = cv2.absdiff(prev_img, curr_img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contours_with_area = [
            (c, cv2.contourArea(c))
            for c in contours
            if cv2.contourArea(c) >= self.MIN_CONTOUR_AREA
        ]
        contours_with_area.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in contours_with_area]

    def extract_click_points(self, contours) -> list[Point]:
        points: list[Point] = []

        for contour in contours:
            if len(points) >= self.MAX_CLICKS_PER_LOOP:
                return points

            point = self.get_point(contour)
            if point is None:
                continue

            if self.check_distance(point, points):
                continue

            points.append(point)

        return points

    def get_point(self, contour) -> Optional[Point]:
        match self._contour_click_mode:
            case ContourClickMode.RANDOM:
                bx, by, bw, bh = cv2.boundingRect(contour)

                for _ in range(5):
                    x = random.randint(bx, bx + bw)
                    y = random.randint(by, by + bh)
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        return Point(x, y)

            case ContourClickMode.CENTER:
                m = cv2.moments(contour)
                if m["m00"] == 0:
                    return None
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])
                return Point(x, y)

        return None

    async def process_clicks(self, points: list[Point]) -> list[Point]:
        result = []

        pydirectinput.keyDown("tab")
        await asyncio.sleep(0.3)

        for point in points:
            collect_point = self.collect_mouse_move(point, self._origin)

            await self.click_point(collect_point)
            await self.return_origin()

            result.append(collect_point)

        pydirectinput.keyUp("tab")

        return result

    def draw_debug(self, contours, points: list[Point], result: list[Point], curr_img):
        output = curr_img.copy()

        for p in points:
            cv2.drawMarker(
                output,
                (p.x, p.y),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
        for p in result:
            cv2.drawMarker(
                output,
                (p.x - self._window.left, p.y - self._window.top),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        cv2.imwrite("output.png", output)

    async def run(self):
        self._task = asyncio.create_task(self.main_loop())
        self._task2 = asyncio.create_task(self.keyboard_monitor_task())
        await asyncio.gather(self._task, self._task2)

    async def click_point(self, point: Point):
        pydirectinput.moveTo(point.x, point.y)
        await asyncio.sleep(0.1)
        pydirectinput.mouseDown()
        await asyncio.sleep(0.02)
        pydirectinput.mouseUp()

    async def return_origin(self):
        if self._origin is None:
            return
        pydirectinput.moveTo(self._origin.x, self._origin.y)
        await asyncio.sleep(0.1)

    def check_distance(self, cp: Point, points: list[Point]) -> bool:
        return any(
            ((cp.x - point.x) ** 2 + (cp.y - point.y) ** 2) ** 0.5
            < self.MIN_CLICK_DISTANCE
            for point in points
        )

    def collect_mouse_move(self, target: Point, origin: Point) -> Point:
        MOUSE_MOVE_DIVISOR = 2.0
        EDGE_SUPPRESSION_FACTOR = 0.2

        abs_point = Point(self._window.left + target.x, self._window.top + target.y)

        half_width = self._window.width / 2

        dist_x = abs(target.x - half_width)
        norm_x = dist_x / half_width
        suppression = max(0, 1.0 - (norm_x * EDGE_SUPPRESSION_FACTOR))

        dx = abs_point.x - origin.x
        dy = abs_point.y - origin.y

        move_x = int(origin.x + (dx * suppression) / MOUSE_MOVE_DIVISOR)
        move_y = int(origin.y + dy / MOUSE_MOVE_DIVISOR)

        return Point(move_x, move_y)

    def is_ready(self) -> bool:
        return self._origin is not None and self._window is not None

    async def sleep(self):
        await asyncio.sleep(self.LOOP_WAIT_TIME)

if __name__ == "__main__":
    asyncio.run(Image().run())
