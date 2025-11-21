import logging
from typing import Optional, Tuple

import pygetwindow
from pygetwindow._pygetwindow_win import Win32Window


class WindowController:
    def __init__(self, target_size: Tuple[int, int]):
        self._window: Optional[Win32Window] = None
        self._original_size: Optional[Tuple[int, int]] = None
        self._target_size = target_size

    @property
    def window(self) -> Optional[Win32Window]:
        return self._window

    @property
    def region(self) -> Optional[list[int]]:
        if self._window is None:
            return None

        return [
            self._window.left,
            self._window.top,
            self._window.width,
            self._window.height,
        ]

    def set_window(self) -> bool:
        self.restore()

        new_window = pygetwindow.getActiveWindow()
        if new_window is None:
            logging.warning("Active window not found.")
            return False

        self._window = new_window
        self._original_size = (new_window.width, new_window.height)

        logging.info(f"Window setted :{self.window.title}")
        return True

    def resize(self) -> bool:
        try:
            self.window.resizeTo(*self._target_size)
            return True
        except Exception as e:
            logging.warning(f"Resize failed: {e}")
            return False

    def restore(self) -> None:
        if self._window is None or self._original_size is None:
            return

        try:
            w, h = self._original_size
            self._window.resizeTo(w, h)
            logging.info("Window size restored.")
        except Exception as e:
            logging.warning(f"Restore failed: {e}")
