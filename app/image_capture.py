import logging
import asyncio
from typing import Optional, Tuple

import cv2
import pyautogui
import numpy as np

from app.window_controroller import WindowController


class ImageCapture:
    def __init__(self, window_controller: WindowController):
        self.window_controller = window_controller

    async def capture_pair(
        self, delay: float = 0.1
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        region = self.window_controller.region
        if region is None:
            logging.warning("Capture failed: window not set.")
            return None, None

        try:
            img1 = await self._shot(region)
            await asyncio.sleep(delay)
            img2 = await self._shot(region)
        except Exception as e:
            logging.warning(f"Screenshot failed: {e}")
            return None, None

        return img1, img2

    async def _shot(self, region) -> np.ndarray:
        pil_img = await asyncio.to_thread(pyautogui.screenshot, None, region)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv_img
