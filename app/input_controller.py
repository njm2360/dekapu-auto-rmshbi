import asyncio
import pyautogui
import pydirectinput
from typing import Optional

from app.window_controroller import WindowController
from app.point import Point


class InputController:
    def __init__(
        self,
        window_controller: WindowController,
        click_down_delay: float = 0.1,
        click_up_delay: float = 0.05,
        mouse_move_divisor: float = 2.0,
        edge_suppression_factor: float = 0.2,
    ):
        self._window_controller = window_controller

        self.click_down_delay = click_down_delay
        self.click_up_delay = click_up_delay

        self._mouse_move_divisor = mouse_move_divisor
        self._edge_suppression_factor = edge_suppression_factor

        self._lock = asyncio.Lock()
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
                    await asyncio.sleep(0.1)
                    await self.click()

            await self.move_to(self._origin)

    async def move_to(self, point: Point):
        await asyncio.to_thread(pydirectinput.moveTo, point.x, point.y)

    async def click(self):
        await asyncio.to_thread(pydirectinput.mouseDown)
        await asyncio.sleep(self.click_up_delay)
        await asyncio.to_thread(pydirectinput.mouseUp)
        await asyncio.sleep(self.click_down_delay)

    def perspective_lock(self):
        pydirectinput.keyDown("tab")
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

        half_width = window.width / 2

        dist_x = abs(point.x - half_width)
        norm_x = dist_x / half_width
        suppression = max(0.0, 1.0 - (norm_x * self._edge_suppression_factor))

        delta = abs_point - self._origin

        move = self._origin + Point(
            x=int(delta.x * suppression / self._mouse_move_divisor),
            y=int(delta.y / self._mouse_move_divisor),
        )

        return move
