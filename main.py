import logging
import asyncio
import keyboard
from typing import Final, Optional

from app.mask_loader import MaskLoader
from app.window_controroller import WindowController
from app.image_capture import ImageCapture
from app.motion_detector import MotionDetector
from app.input_controller import InputController
from app.clickpoint_extractor import ClickPointExtractor


logging.basicConfig(level=logging.INFO)


class AutoClickApp:

    WINDOW_SIZE: Final[tuple[int, int]] = (1024, 768)

    TERMINATE_KEY: Final[str] = "esc"
    SET_WINDOW_KEY: Final[str] = "f5"
    TOGGLE_ENABLE_KEY: Final[str] = "f6"

    LOOP_WAIT_TIME: Final[int] = 3
    DRY_RUN: Final[bool] = False

    def __init__(self):

        self.window_ctrl = WindowController(self.WINDOW_SIZE)
        self.input_ctrl = InputController(window_controller=self.window_ctrl)
        self.capture = ImageCapture(self.window_ctrl)
        self.point_extractor = ClickPointExtractor()

        mask = MaskLoader.make_default(size=self.WINDOW_SIZE)
        self.detector = MotionDetector(mask)

        self._main_task: Optional[asyncio.Task] = None
        self._auto_click_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        keyboard.on_press_key(self.TERMINATE_KEY, self.terminate)
        keyboard.on_press_key(self.SET_WINDOW_KEY, self.set_window)
        keyboard.on_press_key(self.TOGGLE_ENABLE_KEY, self.toggle_run)

    def terminate(self, _):
        logging.info("ESC pressed: stop")

        self._running = False

        if self._main_task:
            self._main_task.cancel()
        if self._auto_click_task:
            self._auto_click_task.cancel()

    def set_window(self, _):
        if self._running:
            logging.warning("Ignored set window (now running)")
            return

        if self.window_ctrl.set_window():
            self.window_ctrl.resize()

    def toggle_run(self, _):
        if self._running:
            self._running = False
            return

        if self.window_ctrl.window is None:
            logging.warning("Window is not set")
            return

        self._running = True

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self.input_ctrl.perspective_lock(), self._loop
            )

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._main_task = asyncio.create_task(self.main())
        self._auto_click_task = asyncio.create_task(self.auto_click_loop())
        await asyncio.gather(self._main_task, self._auto_click_task)

    async def main(self):
        try:
            while True:
                if self._running:
                    await self.loop_step()
                    await asyncio.sleep(self.LOOP_WAIT_TIME)
                else:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self.input_ctrl.cleanup()
            self.window_ctrl.restore()

    async def auto_click_loop(self):
        try:
            while True:
                await asyncio.sleep(0.2)
                if self._running:
                    if not self.input_ctrl.lock.locked():
                        await self.input_ctrl.click()
        except asyncio.CancelledError:
            pass

    async def loop_step(self):
        prev_img, curr_img = await self.capture.capture_pair()
        if prev_img is None or curr_img is None:
            return

        contours = self.detector.detect(prev_img, curr_img)
        if not contours:
            return

        click_points = self.point_extractor.extract(contours)
        if not click_points:
            return

        await self.input_ctrl.execute_clicks(click_points, dry_run=self.DRY_RUN)

        # self.detector.debug(contours, curr_img)


if __name__ == "__main__":
    try:
        asyncio.run(AutoClickApp().run())
    except KeyboardInterrupt:
        pass
