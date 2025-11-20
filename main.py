import cv2
import math
import random
import logging
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


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, factor: float) -> "Point":
        return Point(int(self.x * factor), int(self.y * factor))

    def offset(self, dx: int = 0, dy: int = 0) -> "Point":
        return Point(self.x + dx, self.y + dy)

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def move_to(self):
        pydirectinput.moveTo(self.x, self.y)


class ContourClickMode(IntEnum):
    CENTER = 0
    RANDOM = 1


class Image:
    THRESHOLD_VALUE: Final[int] = 30
    LOOP_WAIT_TIME: Final[int] = 5
    MIN_CONTOUR_AREA: Final[int] = 200
    MIN_CLICK_DISTANCE: Final[int] = 20
    MAX_CLICKS_PER_LOOP: Final[int] = 10
    RANDOM_CLICKS_PER_CONTOUR: Final[int] = 3

    WINDOW_SIZE: Final[tuple[int, int]] = (1024, 768)

    ENABLE_AVOID_CLOSE_CLICK: Final[bool] = True
    DRY_RUN: Final[bool] = False

    TERMINATE_KEY: Final[str] = "esc"
    SET_WINDOW_KEY: Final[str] = "f5"
    TOGGLE_ENABLE_KEY: Final[str] = "f6"

    def __init__(self, contour_click_mode: ContourClickMode = ContourClickMode.RANDOM):
        self._main_task: asyncio.Task = None
        self._auto_click_task: asyncio.Task = None
        self._click_lock = asyncio.Lock()
        self._running: bool = False

        self._window: Optional[Win32Window] = None
        self._original_size: Optional[tuple[int, int]] = None

        self._mask: Optional[np.ndarray] = self._load_mask("mask.png")

        width, height = pyautogui.size()
        self._screen_width = width
        self._screen_height = height
        logging.debug(f"Screen size : {width}x{height}")

        self._contour_click_mode = contour_click_mode

        keyboard.on_press_key(self.TERMINATE_KEY, self.terminate)
        keyboard.on_press_key(self.SET_WINDOW_KEY, self.set_window)
        keyboard.on_press_key(self.TOGGLE_ENABLE_KEY, self.toggle_run)

    def terminate(self, _):
        logging.info("ESC pressed: Emergency stop")

        self._running = False

        if self._main_task:
            self._main_task.cancel()
        if self._auto_click_task:
            self._auto_click_task.cancel()

    def set_window(self, _):
        if self._running:
            logging.warning("Ignored set window (now running)")
            return
        window = pygetwindow.getActiveWindow()
        if window is None:
            logging.warning("Active window not found.")
            return

        self.restore_window_size()
        self._window = window
        self._original_size = (window.width, window.height)

        window.resizeTo(*self.WINDOW_SIZE)
        logging.info(f"Window : {window}")

    def toggle_run(self, _):
        if self._running:
            self._running = False
            logging.info("Stop")
            return

        if self._window is None:
            logging.warning("Window is not set.")
            return

        self._running = True

    def restore_window_size(self):
        if self._window is None or self._original_size is None:
            return

        try:
            w, h = self._original_size
            self._window.resizeTo(w, h)
            logging.info("Window size restored.")
        except Exception as e:
            logging.warning(f"Restore failed: {e}")

    async def run(self):
        self._main_task = asyncio.create_task(self.main_loop())
        self._auto_click_task = asyncio.create_task(self.auto_click_loop())
        await asyncio.gather(self._main_task, self._auto_click_task)

    async def main_loop(self):
        try:
            while True:
                if self._running:
                    await self.loop_step()
                    await asyncio.sleep(self.LOOP_WAIT_TIME)
        except asyncio.CancelledError:
            pass
        finally:
            pydirectinput.mouseUp()
            pydirectinput.keyUp("tab")
            self.restore_window_size()

    async def auto_click_loop(self):
        try:
            while True:
                await asyncio.sleep(0.20)
                if self._running:
                    if not self._click_lock.locked():
                        pydirectinput.mouseDown()
                        await asyncio.sleep(0.05)
                        pydirectinput.mouseUp()

        except asyncio.CancelledError:
            pass

    async def loop_step(self):
        prev_img, curr_img = await self.capture_window()
        if prev_img is None or curr_img is None:
            return

        contours = self.get_contours(prev_img, curr_img)
        if not contours:
            return

        points = self.extract_click_points(contours)
        if not points:
            return

        await self.process_clicks(points)

        self.draw_debug(contours, points, curr_img)

    async def capture_window(self):
        if self._window is None:
            return

        region = [
            self._window.left,
            self._window.top,
            self._window.width,
            self._window.height,
        ]

        try:
            img1 = pyautogui.screenshot(region=region)
            await asyncio.sleep(0.1)
            img2 = pyautogui.screenshot(region=region)
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

        if self._mask is not None:
            dilated = cv2.bitwise_and(dilated, self._mask)

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
        mode = self._contour_click_mode

        for contour in contours:
            if len(points) >= self.MAX_CLICKS_PER_LOOP:
                return points

            if mode == ContourClickMode.RANDOM:
                candidates = (
                    self.get_point(contour, mode)
                    for _ in range(self.RANDOM_CLICKS_PER_CONTOUR)
                )
            elif mode == ContourClickMode.CENTER:
                candidates = (self.get_point(contour, mode),)
            else:
                continue

            for point in candidates:
                if point is None:
                    continue
                if self.ENABLE_AVOID_CLOSE_CLICK:
                    if any(
                        point.distance_to(p) < self.MIN_CLICK_DISTANCE for p in points
                    ):
                        continue
                points.append(point)

        return points

    def get_point(self, contour, mode: ContourClickMode) -> Optional[Point]:
        match mode:
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

    async def process_clicks(self, points: list[Point]):
        async with self._click_lock:
            pydirectinput.keyDown("tab")
            await asyncio.sleep(0.2)

            p = pyautogui.position()
            origin = Point(p.x, p.y)

            for point in points:
                correct_point = self.correct_mouse_move(point, origin)
                correct_point.move_to()

                if self.DRY_RUN:
                    await asyncio.sleep(1)
                else:
                    await asyncio.sleep(0.1)
                    pydirectinput.mouseDown()
                    await asyncio.sleep(0.05)
                    pydirectinput.mouseUp()

            origin.move_to()
            await asyncio.sleep(0.2)
            pydirectinput.keyUp("tab")

    def correct_mouse_move(self, point: Point, origin: Point) -> Point:
        MOUSE_MOVE_DIVISOR = 2.0
        EDGE_SUPPRESSION_FACTOR = 0.2

        abs_point = Point(self._window.left, self._window.top) + point

        half_width = self._window.width / 2  # Maybe screen width / 2 ?

        dist_x = abs(point.x - half_width)
        norm_x = dist_x / half_width
        suppression = max(0, 1.0 - (norm_x * EDGE_SUPPRESSION_FACTOR))

        delta = abs_point - origin

        move = origin + Point(
            x=int(delta.x * suppression / MOUSE_MOVE_DIVISOR),
            y=int(delta.y / MOUSE_MOVE_DIVISOR),
        )

        return move

    def draw_debug(self, contours, points: list[Point], curr_img):
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

    def _load_mask(self, mask_path: Optional[str] = None) -> np.ndarray:
        if mask_path is None:
            logging.info("Mask path not provided. Using default mask")
            return self._create_default_mask()

        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            logging.warning(f"Failed to load mask image. Using default mask")
            return self._create_default_mask()

        h, w = mask_gray.shape[:2]

        if (w, h) != self.WINDOW_SIZE:
            logging.warning(
                f"Mask size {w}x{h} does not match WINDOW_SIZE {self.WINDOW_SIZE}. "
                "Using default mask"
            )
            return self._create_default_mask()

        _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        logging.info("Mask image load successfully")
        return mask_bin

    def _create_default_mask(self) -> np.ndarray:
        w, h = self.WINDOW_SIZE
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2 :, :] = 255
        return mask


if __name__ == "__main__":
    try:
        asyncio.run(Image().run())
    except KeyboardInterrupt:
        pass
