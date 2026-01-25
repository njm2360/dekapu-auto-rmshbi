import asyncio
import pyautogui
import pydirectinput
from typing import Final, Optional

from app.window_controroller import WindowController
from app.point import Point


class InputController:

    MOUSE_TAKE_WAIT: Final[float] = 0.10
    MOVE_AFTR_WAIT: Final[float] = 0.10
    CLICK_DOWN_WAIT: Final[float] = 0.10
    MOUSE_MOVE_DIVISOR: Final[float] = 2.0
    FOV_SUPPRESSION_FACTOR: Final[float] = 0.2

    def __init__(self, window_controller: WindowController):
        self._window_controller = window_controller

        self._lock: asyncio.Lock = asyncio.Lock()
        self._origin: Optional[Point] = None

    @property
    def lock(self):
        return self._lock

    async def execute_clicks(self, points: list[Point], dry_run: bool = False):
        if self._origin is None:
            raise RuntimeError("Origin is not set")
        if not points:
            return

        corrected = [self.correct(p) for p in points]

        async with self._lock:
            for point in corrected:
                await self.move_to(point)

                if dry_run:
                    await asyncio.sleep(1)
                else:
                    await asyncio.sleep(self.MOVE_AFTR_WAIT)
                    await self.click()

            await self.move_to(self._origin)

    async def move_to(self, point: Point):
        pydirectinput.moveTo(point.x, point.y)

    async def click(self):
        pydirectinput.mouseDown()
        await asyncio.sleep(self.CLICK_DOWN_WAIT)
        pydirectinput.mouseUp()

    async def perspective_lock(self):
        pydirectinput.keyDown("tab")
        await asyncio.sleep(self.MOUSE_TAKE_WAIT)
        p = pyautogui.position()
        self._origin = Point(p.x, p.y)

    def cleanup(self):
        pydirectinput.mouseUp()
        pydirectinput.keyUp("tab")

    def correct(self, point: Point) -> Point:
        window = self._window_controller.window
        if window is None:
            raise RuntimeError("Window is not set")
        if self._origin is None:
            raise RuntimeError("Origin is not set")

        abs_point = Point(window.left, window.top) + point

        half_width = window.width / 2.0

        dist_x = abs(point.x - half_width)
        norm_x = dist_x / half_width
        suppression = max(0.0, 1.0 - (norm_x * self.FOV_SUPPRESSION_FACTOR))

        delta = abs_point - self._origin

        move = self._origin + Point(
            x=int(delta.x * suppression / self.MOUSE_MOVE_DIVISOR),
            y=int(delta.y / self.MOUSE_MOVE_DIVISOR),
        )

        return move
